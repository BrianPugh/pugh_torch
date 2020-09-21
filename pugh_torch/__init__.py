# -*- coding: utf-8 -*-

"""Top-level package for Pugh Torch."""

__author__ = "Brian Pugh"
__email__ = "bnp117@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.3.0"


def get_module_version():
    return __version__


from .exceptions import *

import pugh_torch.datasets
import pugh_torch.helpers
import pugh_torch.losses
import pugh_torch.mappings
import pugh_torch.modules
import pugh_torch.transforms
import pugh_torch.augmentations
import pugh_torch.augmentations as A
