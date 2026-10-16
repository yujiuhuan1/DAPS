from abc import abstractmethod
from argparse import Namespace
from torch import nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
import torch.optim
from backbone.gnn.gcn import GCN_Net
from backbone.gnn.pmlp import PMLP_GCN


class FederatedDataset:
    NAME = None
    SETTING = None
    N_SAMPLES_PER_Class = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        self.train_loaders = []
        self.test_loader = []
        self.args = args

    def get_gnn_backbone_dict(self):
        nets_dict = {'gcn': GCN_Net, 'pmlp_gcn': PMLP_GCN}
        return nets_dict

    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num, names_list) -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        pass

    def get_parti(self):
        count= 0
        for key in self.domain_dict.keys():
            count += self.domain_dict[key]
        return count
    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass
