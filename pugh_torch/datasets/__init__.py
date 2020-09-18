"""
pugh_torch.datasets.__init__

The root dataset path can be set via the environmental variable
``PUGH_TORCH_DATASET_PATH``.

I don't expose this in code because I think it just clutters the code.
"""

import os
from pathlib import Path

ROOT_DATASET_PATH = Path(
    os.environ.get("PUGH_TORCH_DATASET_PATH", "~/.pugh_torch/datasets")
).expanduser()

# Populated automatically via pugh_torch.datasets.Dataset.__init_subclass__
DATASETS = {}

from .base import Dataset
from .torchvision import TorchVisionDataset

import pugh_torch.datasets.classification
import pugh_torch.datasets.segmentation


def get(genre, name):
    """Gets dataset constructor from string identifiers

    Parameters
    ----------
    genre : str
        Type of dataset. e.x. "classification".
        Case insensitive
    name : str
        Name of dataset. e.x. "imagenet".
        Case insensitive
    """

    genre = genre.lower()
    name = name.lower()

    return DATASETS[genre][name]


# Alias for ``get`` function
get_dataset = get
