from .twitch import FedTwitch

Priv_NAMES = {'fl_twitch': FedTwitch}


def get_prive_dataset(args):
    return Priv_NAMES[args.dataset](args)
