import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import random
import torch
import numpy as np


def get_device(device_id) -> torch.device:
    return torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")


def data_path() -> str:
    return str(PROJECT_ROOT / 'datasets' / 'data') + os.sep


def base_path() -> str:
    return str(PROJECT_ROOT / 'data') + os.sep


def checkpoint_path() -> str:
    return str(PROJECT_ROOT / 'checkpoint') + os.sep


def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.
    :param seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
