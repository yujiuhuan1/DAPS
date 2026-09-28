from argparse import Namespace

from datasets.twitch import FedTwitch

Priv_NAMES = {FedTwitch.NAME: FedTwitch}
Pub_NAMES = {}


def get_prive_dataset(args: Namespace) -> FedTwitch:
    if args.dataset != FedTwitch.NAME:
        raise ValueError(f"This release supports only {FedTwitch.NAME!r}.")
    return FedTwitch(args)

