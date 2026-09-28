import os
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Twitch

from backbone.gnn.pmlp import PMLP_GCN
from datasets.utils.federated_dataset import FederatedDataset
from datasets.utils.splitter import RandomSplitter
from utils.conf import data_path


class Twitch1(Twitch):
    def __init__(self, root: str, name: str, transform=None, pre_transform=None):
        super().__init__(root, name, transform, pre_transform)
        self.data_name = name
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def process(self):
        raw = np.load(self.raw_paths[0], "r", allow_pickle=True)
        x = torch.from_numpy(raw["features"]).to(torch.float)
        y = torch.from_numpy(raw["target"]).to(torch.long)
        edge_index = torch.from_numpy(raw["edges"]).to(torch.long).t().contiguous()
        graph = Data(x=x, y=y, edge_index=edge_index)

        indices = np.random.permutation(y.shape[0])
        train_end = int(len(indices) * 0.6)
        val_end = train_end + int(len(indices) * 0.1)
        for name, selected in (
            ("train_mask", indices[:train_end]),
            ("val_mask", indices[train_end:val_end]),
            ("test_mask", indices[val_end:]),
        ):
            mask = torch.zeros(y.shape[0], dtype=torch.bool)
            mask[selected] = True
            setattr(graph, name, mask)

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        torch.save(self.collate([graph]), self.processed_paths[0])


class FedTwitch(FederatedDataset):
    NAME = "fl_twitch"
    SETTING = "domain_skew"
    DOMAINS_LIST = ["DE", "EN", "ES", "FR", "PT", "RU"]
    # Paper alpha=5: 10 German clients and 2 clients for each other domain.
    domain_dict = {"EN": 2, "ES": 2, "FR": 2, "PT": 2, "RU": 2, "DE": 10}
    N_CLASS = 2

    def get_data_loaders(self, selected_domain_list=None):
        train_graphs = []
        test_graphs = []
        for domain, client_count in (selected_domain_list or {}).items():
            splitter = RandomSplitter(client_count)
            root = os.path.join(data_path(), "Twitch")
            local_dataset = Twitch1(root=root, name=domain)
            global_dataset = Twitch1(root=root, name=domain)
            graphs, global_graph = splitter(local_dataset[0], global_dataset[0], percent=30)
            train_graphs.extend(graphs)
            test_graphs.append(global_graph)
        return train_graphs, test_graphs

    def get_backbone(self, parti_num, names_list=None):
        return [PMLP_GCN(128, self.N_CLASS, self.args.hidden) for _ in range(parti_num)]

