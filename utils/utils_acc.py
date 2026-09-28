import torch
from sklearn.metrics import accuracy_score


def get_acc_score(pred, label):
    predicted = pred.detach().clone().argmax(dim=1)
    return accuracy_score(label.cpu().detach().numpy(), predicted.cpu().detach().numpy())


def get_scores(pred, label, name=""):
    accuracy = get_acc_score(pred, label)
    print(f"{name} result:")
    print(f"acc score: {accuracy:.4f}\n")
    return accuracy, 1

