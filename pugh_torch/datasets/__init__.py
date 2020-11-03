"""
pugh_torch.datasets.__init__

The root dataset path can be set via the environmental variable
``PUGH_TORCH_DATASETS_PATH``.

I don't expose this in code because I think it just clutters the code.
"""

import os
from pathlib import Path

ROOT_DATASET_PATH = Path(
    os.environ.get("PUGH_TORCH_DATASETS_PATH", "~/.pugh_torch/datasets")
).expanduser()

# Populated automatically via pugh_torch.datasets.Dataset.__init_subclass__
DATASETS = {}

from .base import Dataset
from .torchvision import TorchVisionDataset

import pugh_torch.datasets.classification
import pugh_torch.datasets.segmentation


def get(*args):
    """Gets dataset constructor from string identifiers

    Example:
        constructor = get("classification", "imagenet")

    Parameters
    ----------
    *args : str
        Case-insensitive Strings that lead to a dataset.
        Typically in form ``(genre, name)``
        Type of dataset. e.x. "classification".
    """

    d = DATASETS
    for arg in args:
        d = d[arg.lower()]

    assert issubclass(
        d, Dataset
    ), f"arguments {args} did not lead to a dataset constructor; lead to {d}"

    return d


# Alias for ``get`` function
get_dataset = get

from .nyuv2 import NYUv2
