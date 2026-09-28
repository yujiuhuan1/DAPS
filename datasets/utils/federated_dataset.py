from argparse import Namespace


class FederatedDataset:
    NAME = None
    SETTING = None
    N_CLASS = None
    DOMAINS_LIST = []
    domain_dict = {}

    def __init__(self, args: Namespace) -> None:
        self.args = args

    def get_parti(self):
        return sum(self.domain_dict.values())

    def get_data_loaders(self, selected_domain_list=None):
        raise NotImplementedError

    def get_backbone(self, parti_num, names_list=None):
        raise NotImplementedError

    def get_transform(self):
        return None

