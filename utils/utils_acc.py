import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score


def get_acc_score(pred, label):
    pred_label = pred.detach().clone()
    pred_label = pred_label.argmax(dim=1)
    acc_score = accuracy_score(label.cpu().detach().numpy(), pred_label.cpu().detach().numpy())
    return acc_score


def get_auc_score(pred, label):
    try:
        pred = F.softmax(pred, dim=1)
        #print(torch.unique(label).size(0))
        if torch.unique(label).size(0) > 2:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), multi_class='ovr')
        else:
            auc_score = roc_auc_score(label.cpu().detach().numpy(), pred[:, 1].cpu().detach().numpy())
        return auc_score
    except ValueError:
        print("auc error")
        return None
        pass


def get_scores(pred, label, name=''):
    # auc_score = get_auc_score(pred, label)
    auc_score = 1
    acc_score = get_acc_score(pred, label)
    print(name + ' result:')
    print("acc score: " + str(round(acc_score,4)) )
    # print("acc score: " + str(round(acc_score,3)) + " auc score: " + str(round(auc_score,3)) )
    print()


    return acc_score, auc_score
