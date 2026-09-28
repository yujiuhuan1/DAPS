import itertools

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import copy
from scipy.sparse import coo_matrix

from backbone.gnn.mlp import Linear
from backbone.knn import MomentumQueue
from utils.args import *
import numpy as np
import torch.nn.functional as F
from utils.finch import FINCH
import torch_geometric
from models.utils.federated_model import FederatedModel
from utils.utils_acc import get_scores
from sklearn.neighbors import kneighbors_graph
from utils.util import diff_loss, proto_align_loss, get_stable_node, dict_to_tensor, soft_predict, \
    edge_index_to_adj_matrix, get_norm_and_orig


def get_proto_norm_weighted(num_classes, embedding, class_label, weight,unique_labels):
    m1= F.one_hot(class_label, num_classes=num_classes)
    m2 = (m1 * weight[:, None]).t()
    m = m2 / (m2.sum(dim=1, keepdim=True)+ 1e-6)
    m = m[unique_labels]
    return torch.mm(m, embedding)


def calculate_anchor_drift(previous_anchors, current_anchors):
    """Average final-anchor drift over classes present in consecutive rounds."""
    common_labels = sorted(set(previous_anchors) & set(current_anchors))
    if not common_labels:
        return float('nan')

    class_drifts = []
    for label in common_labels:
        previous = previous_anchors[label].detach().float().reshape(1, -1)
        current = current_anchors[label].detach().float().reshape(1, -1)
        previous = previous.to(current.device)
        cosine = F.cosine_similarity(current, previous, dim=1).clamp(-1.0, 1.0)
        class_drifts.append(1.0 - cosine.item())

    return float(np.mean(class_drifts))


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def com_distillation_loss(t_logits, s_logits, adj_orig, adj_sampled, temp, loss_mode):

    s_dist = F.log_softmax(s_logits / temp, dim=-1)
    t_dist = F.softmax(t_logits / temp, dim=-1)
    if loss_mode == 0:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist)
    elif loss_mode == 1:
        kd_loss = temp * temp * F.kl_div(s_dist, t_dist.detach())

    adj = torch.triu(adj_orig * adj_sampled).detach()
    edge_list = (adj + adj.T).nonzero().t()

    s_dist_neigh = F.log_softmax(s_logits[edge_list[0]] / temp, dim=-1)
    t_dist_neigh = F.softmax(t_logits[edge_list[1]] / temp, dim=-1)
    if loss_mode == 0:
        kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh)
    elif loss_mode == 1:
        kd_loss += temp * temp * F.kl_div(s_dist_neigh, t_dist_neigh.detach())

    return kd_loss

class fggp(FederatedModel):
    NAME = 'fggp'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list,args, transform):
        super(fggp, self).__init__(nets_list,args,transform)
        self.global_centroids = []
        self.local_centroids = {}
        self.global_protos = []
        self.local_protos = {}
        self.local_proto_stats = {}
        self.global_proto_weights = {}
        self.global_anchor_protos = {}
        self.global_anchor_quality = {}
        self.local_protos_ema = {}
        self.anchor_drift_history = []
        self.infoNCET = args.infoNCET
        self.eval = args.size

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        self.personal_project = [Linear(self.global_net.hidden_channels, self.global_net.hidden_channels, 0.5, bias=True) for _ in range(len(self.nets_list))]
        self.eval_knn = MomentumQueue(self.global_net.hidden_channels,
                                           self.N_CLASS * self.eval, 0.1, self.args.knn,
                                           self.N_CLASS).to(self.device)
        for _,net in enumerate(self.nets_list):
            net.load_state_dict(global_w)


    def _as_proto2d(self, proto):
        if not torch.is_tensor(proto):
            proto = torch.tensor(proto, dtype=torch.float32)
        proto = proto.detach().float()
        if proto.dim() == 1:
            proto = proto.view(1, -1)
        return proto

    def _client_proto_weight(self, client_idx, label):
        stats = self.local_proto_stats.get(client_idx, {}).get(int(label), None)
        if stats is None:
            return 1.0
        support = max(float(stats.get('support', 1.0)), 1.0)
        confidence = float(stats.get('confidence', 1.0))
        return max(confidence * np.log1p(support), 1e-3)

    def _weighted_cluster_mean(self, proto_array, weights):
        weights = np.asarray(weights, dtype=np.float32)
        if weights.sum() <= 1e-12:
            return np.mean(proto_array, axis=0, keepdims=True)
        return np.average(proto_array, axis=0, weights=weights).reshape(1, -1)

    def _record_anchor_drift(self, previous_anchors):
        anchor_drift = calculate_anchor_drift(
            previous_anchors,
            self.global_anchor_protos,
        )
        self.anchor_drift_history.append(anchor_drift)
        if np.isnan(anchor_drift):
            drift_text = 'nan'
        else:
            drift_text = f'{anchor_drift:.8f}'
        print(f'Anchor Drift Round {self.epoch_index}: {drift_text}')

    def _update_anchor_bank(self, agg_protos_label):
        previous_anchors = {
            int(label): anchor.detach().clone()
            for label, anchor in self.global_anchor_protos.items()
        }
        momentum = float(getattr(self.args, 'daps_momentum', getattr(self.args, 'ema', 0.9)))
        warmup = int(getattr(self.args, 'daps_warmup', 5))
        drift_temp = max(float(getattr(self.args, 'daps_drift_temp', 0.25)), 1e-6)

        for label, proto_list in agg_protos_label.items():
            label = int(label)
            if len(proto_list) == 0:
                continue
            weights = self.global_proto_weights.get(label, [1.0 for _ in proto_list])
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            weights_tensor = weights_tensor / (weights_tensor.sum() + 1e-12)
            proto_tensor = torch.cat([self._as_proto2d(proto).to(self.device) for proto in proto_list], dim=0)
            new_anchor = torch.sum(proto_tensor * weights_tensor.view(-1, 1), dim=0)
            new_anchor = F.normalize(new_anchor.view(1, -1), dim=1).squeeze(0)

            old_anchor = self.global_anchor_protos.get(label, None)
            if old_anchor is None or self.epoch_index < warmup:
                anchor = new_anchor
                quality = 1.0
            else:
                old_anchor = F.normalize(old_anchor.to(self.device).view(1, -1), dim=1).squeeze(0)
                cosine = F.cosine_similarity(new_anchor.view(1, -1), old_anchor.view(1, -1)).clamp(-1.0, 1.0).item()
                drift = max(0.0, 1.0 - cosine)
                drift_gate = float(np.exp(-drift / drift_temp))
                raw_quality = float(np.mean(weights))
                quality = raw_quality / (raw_quality + 1.0)
                update_rate = (1.0 - momentum) * (0.25 + 0.75 * quality * drift_gate)
                update_rate = min(max(update_rate, 0.01), 0.5)
                anchor = F.normalize(((1.0 - update_rate) * old_anchor + update_rate * new_anchor).view(1, -1), dim=1).squeeze(0)
                quality = quality * drift_gate

            self.global_anchor_protos[label] = anchor.detach()
            self.global_anchor_quality[label] = quality

        self._record_anchor_drift(previous_anchors)

    def _refresh_eval_queue(self):
        if len(self.global_anchor_protos) == 0:
            return
        queue_size = self.eval_knn.queue_size
        anchors = []
        labels = []
        repeat_num = max(1, int(self.eval))

        for label in sorted(self.global_anchor_protos.keys()):
            anchor = F.normalize(self.global_anchor_protos[label].to(self.device).view(1, -1), dim=1)
            anchors.append(anchor.repeat(repeat_num, 1))
            labels.append(torch.full((repeat_num,), int(label), dtype=torch.long, device=self.device))

        memory = torch.cat(anchors, dim=0)
        memory_label = torch.cat(labels, dim=0)
        if memory.size(0) < queue_size:
            repeat_times = int(np.ceil(queue_size / memory.size(0)))
            memory = memory.repeat(repeat_times, 1)
            memory_label = memory_label.repeat(repeat_times)

        memory = F.normalize(memory[:queue_size], dim=1)
        memory_label = memory_label[:queue_size]
        self.eval_knn.memory.copy_(memory)
        self.eval_knn.memory_label.copy_(memory_label)
        self.eval_knn.index = 0

    def _anchor_tensor(self):
        if len(self.global_anchor_protos) == 0:
            return None, None
        labels = sorted(self.global_anchor_protos.keys())
        anchors = torch.stack([self.global_anchor_protos[label].to(self.device) for label in labels], dim=0)
        anchors = F.normalize(anchors, dim=1)
        return anchors, labels

    def _daps_anchor_loss(self, feat, labels, train_mask, pseudo_labels, confidences):
        anchors, anchor_labels = self._anchor_tensor()
        if anchors is None or len(anchor_labels) < 2:
            return feat.sum() * 0.0

        target_labels = pseudo_labels.detach()
        confidence_threshold = float(getattr(self.args, 'daps_conf', 0.75))
        reliable_mask = torch.logical_or(train_mask, confidences.detach() >= confidence_threshold)
        in_anchor_mask = torch.zeros_like(reliable_mask, dtype=torch.bool)
        for label in anchor_labels:
            in_anchor_mask = torch.logical_or(in_anchor_mask, target_labels == int(label))
        reliable_mask = torch.logical_and(reliable_mask, in_anchor_mask)

        if int(reliable_mask.sum().item()) == 0:
            return feat.sum() * 0.0

        selected_idx = reliable_mask.nonzero(as_tuple=False).view(-1)
        selected_labels = target_labels[selected_idx]
        target = torch.zeros(selected_idx.numel(), dtype=torch.long, device=feat.device)
        for pos, label in enumerate(anchor_labels):
            target[selected_labels == int(label)] = pos

        proto_logits = torch.mm(F.normalize(feat, dim=1), anchors.t())
        proto_logits = proto_logits / max(float(getattr(self.args, 'daps_temp', 0.2)), 1e-6)
        loss = F.cross_entropy(proto_logits[selected_idx], target, reduction='none')

        weights = confidences[selected_idx].detach().clamp(min=0.05)
        weights[train_mask[selected_idx]] = 1.0
        return torch.sum(loss * weights) / (weights.sum() + 1e-12)

    def proto_aggregation(self, local_protos_list):
        agg_protos_label = dict()
        agg_weights_label = dict()
        for idx in self.online_clients:
            if idx not in local_protos_list:
                continue
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                label = int(label)
                proto = self._as_proto2d(local_protos[label])
                weight = self._client_proto_weight(idx, label)
                if label in agg_protos_label:
                    agg_protos_label[label].append(proto)
                    agg_weights_label[label].append(weight)
                else:
                    agg_protos_label[label] = [proto]
                    agg_weights_label[label] = [weight]

        for label, proto_list in list(agg_protos_label.items()):
            weights = np.asarray(agg_weights_label[label], dtype=np.float32)
            proto_array = np.array([item.squeeze(0).detach().cpu().numpy().reshape(-1) for item in proto_list], dtype=np.float32)
            if len(proto_list) > 1:
                c, num_clust, req_c = FINCH(proto_array, initial_rank=None, req_clust=None, distance='cosine',
                                            ensure_early_exit=False, verbose=True)
                class_cluster_array = np.array([c[index, -1] for index in range(c.shape[0])])
                uniqure_cluster = np.unique(class_cluster_array).tolist()
                agg_selected_proto = []
                agg_selected_weight = []

                for _, cluster_index in enumerate(uniqure_cluster):
                    selected_array = np.where(class_cluster_array == cluster_index)[0]
                    selected_proto_list = proto_array[selected_array]
                    selected_weights = weights[selected_array]
                    proto = self._weighted_cluster_mean(selected_proto_list, selected_weights)
                    agg_selected_proto.append(torch.tensor(proto, dtype=torch.float32))
                    agg_selected_weight.append(float(selected_weights.sum()))
                agg_protos_label[label] = agg_selected_proto
                self.global_proto_weights[label] = agg_selected_weight
            else:
                agg_protos_label[label] = [self._as_proto2d(proto_list[0]).cpu()]
                self.global_proto_weights[label] = [float(weights[0])]

        self._update_anchor_bank(agg_protos_label)
        self._refresh_eval_queue()
        return agg_protos_label



    def aggregate_nets(self, freq=None, personal='lkl'):
        global_net = self.global_net
        nets_list = self.nets_list

        online_clients = self.online_clients
        global_w = self.global_net.state_dict()

        if self.args.averaing == 'weight':
            online_clients_dl = [self.trainloaders[online_clients_index] for online_clients_index in online_clients]
            online_clients_len = []
            for dl in online_clients_dl:
                if isinstance(dl, torch_geometric.data.Data):
                    # 判断是否是图数据集
                    online_clients_len.append(dl.num_nodes)
                else:
                    online_clients_len.append(dl.sampler.indices.size)
                    # online_clients_len = [dl.sampler.indices.size if isinstance(dl, torch_geometric.data.Data) eles '2' for dl in online_clients_dl]
            online_clients_all = np.sum(online_clients_len)
            freq = online_clients_len / online_clients_all
        else:
            # if freq == None:
            parti_num = len(online_clients)
            freq = [1 / parti_num for _ in range(parti_num)]

        first = True
        for index, net_id in enumerate(online_clients):
            net = nets_list[net_id]
            net_para = net.state_dict()
            # if net_id == 0:
            if first:
                first = False
                for key in net_para:
                    # 检查网络层的名称是否包含特定名称
                    if personal not in key:
                        global_w[key] = net_para[key] * freq[index]
            else:
                for key in net_para:
                    # 检查网络层的名称是否包含特定名称
                    if personal not in key:
                        global_w[key] += net_para[key] * freq[index]

        global_net.load_state_dict(global_w)

        for _, net in enumerate(nets_list):
            net.load_state_dict(global_net.state_dict())

    def loc_update(self,priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients,self.online_num,replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            self._train_net(i,self.nets_list[i], priloader_list[i])
        # self.global_centroids = self.proto_aggregation(self.local_centroids)
        self.global_protos = self.proto_aggregation(self.local_protos)
        self.aggregate_nets(None,self.args.personal)

        return  None

    def _extract_local_prototypes(self, net, train_loader):
        was_training = net.training
        net.eval()
        with torch.no_grad():
            out = net(train_loader)
            feat = net.features(train_loader)
            output_exp = torch.exp(out)
            confidences = output_exp.max(1)[0]
            pseudo_labels = output_exp.max(1)[1].type_as(train_loader.y)
            pseudo_labels[train_loader.train_mask] = train_loader.y[train_loader.train_mask]
            confidences[train_loader.train_mask] = 1.0
            unique_labels = torch.unique(pseudo_labels)
            proto = get_proto_norm_weighted(self.N_CLASS, feat, pseudo_labels, confidences, unique_labels)

            tensor_dict = {}
            stats_dict = {}
            for proto_index, label in enumerate(unique_labels):
                label_int = int(label.item())
                label_mask = pseudo_labels == label
                support = int(label_mask.sum().item())
                confidence = float(confidences[label_mask].mean().item()) if support > 0 else 0.0
                tensor_dict[label_int] = proto[proto_index].detach().view(1, -1)
                stats_dict[label_int] = {
                    'support': support,
                    'confidence': confidence,
                }

        net.train(was_training)
        return tensor_dict, stats_dict

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        train_loader = train_loader.to(self.device)
        net.train()
        params = itertools.chain(*[net.parameters()])
        optimizer = optim.SGD(params, lr=self.local_lr, momentum=0.9, weight_decay=1e-5)
        criterion = F.nll_loss
        self.global_net = self.global_net.to(self.device)
        # us = get_stable_node(train_loader,self.global_net)
        train_loader2 = self.other_view[index]
        if self.epoch_index % self.args.knn_frequence == 0:
            with torch.no_grad():
                global_feature = self.global_net.features(train_loader)
                adj = kneighbors_graph(global_feature.cpu(), self.args.neibor, metric='cosine')
                del global_feature
                adj.setdiag(1)
                coo = adj.tocoo()
                train_loader.global_edge_index = torch.tensor([coo.row, coo.col], dtype=torch.long).to(self.device)
                del coo
                del adj
                combined_edge_index = torch.cat([train_loader.edge_index, train_loader.global_edge_index], dim=1)
                # 将 edge_index 转换为元组集合，删除重复的边
                # combined_edge_index = torch.cat([train_loader.edge_index, train_loader.edge_index], dim=1)
                edge_set = set(zip(combined_edge_index[0].tolist(), combined_edge_index[1].tolist()))
                # 将结果转换回 edge_index 的格式
                union_edge_index = torch.tensor([[i[0] for i in edge_set], [i[1] for i in edge_set]], dtype=torch.long)
                train_loader2.edge_index = union_edge_index
                train_loader2 = get_norm_and_orig(train_loader2)
                adj_orig = train_loader2.adj_orig
                norm_w = adj_orig.shape[0] ** 2 / float((adj_orig.shape[0] ** 2 - adj_orig.sum()) * 2)
                pos_weight = torch.FloatTensor([float(adj_orig.shape[0] ** 2 - adj_orig.sum()) / adj_orig.sum()])
                train_loader2.norm_w = norm_w
                train_loader2.pos_weight = pos_weight
                train_loader2 = train_loader2.to(self.device)

        iterator = tqdm(range(self.local_epoch))
        for iter in iterator:
            # out = net.(train_loader)
            # out1 = net.conv(train_loader)
            # loss1 = criterion(out1[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            #
            # out2 = net.mlp(train_loader)
            # loss2 = criterion(out2[train_loader.train_mask], train_loader.y[train_loader.train_mask])

            # out3 = net.att_model([out1,out2])
            # loss3 = criterion(out3[train_loader.train_mask], train_loader.y[train_loader.train_mask])

            out4 = net(train_loader)
            # out5 = net(train_loader2)
            adj_sampled, adj_logits = net.aug(train_loader2)
            train_loader2.adj = adj_sampled
            out5 = net(train_loader2, adj=True)
            ga_loss = train_loader2.norm_w * F.binary_cross_entropy_with_logits(adj_logits, train_loader2.adj_orig, pos_weight=train_loader2.pos_weight)
            kd_loss = com_distillation_loss(out4, out5, train_loader2.adj_orig, adj_sampled, 0.1, 0)
            lossCE = criterion(out4[train_loader.train_mask], train_loader.y[train_loader.train_mask])
            lossCE2 = criterion(out5[train_loader2.train_mask], train_loader2.y[train_loader2.train_mask])

            feat = net.features(train_loader)
            feat_global = net.features(train_loader2, adj=True)

            output_exp = torch.exp(out4)
            confidences = output_exp.max(1)[0]
            pseudo_labels = output_exp.max(1)[1].type_as(train_loader.y)
            pseudo_labels[train_loader.train_mask] = train_loader.y[train_loader.train_mask]
            confidences[train_loader.train_mask] = 1.0
            unique_labels = torch.unique(pseudo_labels)
            proto = get_proto_norm_weighted(self.N_CLASS, feat, pseudo_labels, confidences,unique_labels)
            proto_global = get_proto_norm_weighted(self.N_CLASS, feat_global, pseudo_labels, confidences,unique_labels)

            loss_pa = proto_align_loss(proto_global, proto,temperature=0.5)

            if len(self.global_anchor_protos) == 0:
                loss_daps = 0 * lossCE
            else:
                loss_daps = self._daps_anchor_loss(
                    feat, train_loader.y, train_loader.train_mask, pseudo_labels, confidences)
                loss_daps = loss_daps + self._daps_anchor_loss(
                    feat_global, train_loader2.y, train_loader2.train_mask, pseudo_labels, confidences)
                loss_daps = 0.5 * loss_daps

            daps_warmup = max(int(getattr(self.args, 'daps_warmup', 5)), 1)
            daps_ramp = min(1.0, float(self.epoch_index + 1) / float(daps_warmup))
            daps_lambda = float(getattr(self.args, 'daps_lambda', 0.2))
            loss = lossCE + lossCE2 + loss_pa + ga_loss + daps_lambda * daps_ramp * loss_daps

            optimizer.zero_grad()
            loss.backward()
            iterator.desc = "Local Pariticipant %d loss = %0.3f, DAPS = %0.3f" % (index, loss, loss_daps)
            optimizer.step()

        tensor_dict, stats_dict = self._extract_local_prototypes(net, train_loader)
        self.local_protos[index] = tensor_dict
        self.local_proto_stats[index] = stats_dict

