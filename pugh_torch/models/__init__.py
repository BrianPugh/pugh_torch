"""
pugh_torch.models.__init__

The root dataset path can be set via the environmental variable
``PUGH_TORCH_MODELS_PATH``.

I don't expose this in code because I think it just clutters the code.
"""

import os
from pathlib import Path

ROOT_MODELS_PATH = Path(
    os.environ.get("PUGH_TORCH_DATASETS_PATH", "~/.pugh_torch/models")
).expanduser()

import torch
from torch.hub import load_state_dict_from_url

torch.hub.set_dir(str(ROOT_MODELS_PATH))

from .resnet import *
