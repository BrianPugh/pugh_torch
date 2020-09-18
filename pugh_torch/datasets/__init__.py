"""
pugh_torch.datasets.__init__

The root dataset path can be set via the environmental variable
``PUGH_TORCH_DATASET_PATH``.

I don't expose this in code because I think it just clutters the code.
"""

import os
from pathlib import Path

ROOT_DATASET_PATH = Path(
    os.environ.get("PUGH_TORCH_DATASET_PATH", "~/.pugh_torch/data")
)

from .base import Dataset

import pugh_torch.datasets.classification
import pugh_torch.datasets.segmentation


# def get_dataset(name):
#    """
#    Parameters
#    ----------
#    name : str
#    """
#
#    import ipdb as pdb; pdb.set_trace()
#
#    return datasets[name.lower()]
