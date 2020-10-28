# -*- coding: utf-8 -*-

"""Top-level package for Pugh Torch."""

__author__ = "Brian Pugh"
__email__ = "bnp117@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.4.0"


def get_module_version():
    return __version__


from .exceptions import *

import pugh_torch.augmentations
import pugh_torch.augmentations as A
import pugh_torch.callbacks
import pugh_torch.datasets

import pugh_torch.helpers
from pugh_torch.helpers import to_obj

import pugh_torch.linalg
from pugh_torch.linalg import batch_lstsq

import pugh_torch.losses
import pugh_torch.mappings
import pugh_torch.models
import pugh_torch.modules
import pugh_torch.optimizers
import pugh_torch.transforms

import pugh_torch.utils
from pugh_torch.utils import batch_index_select

try:
    import pytorch_lightning
except ImportError:
    pass
else:
    from pugh_torch.modules import LightningModule
