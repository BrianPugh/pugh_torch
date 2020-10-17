import torch
from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief

optimizer_lookup = {
    "adabelief": AdaBelief,
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "asgd": torch.optim.ASGD,
    "lbfgs": torch.optim.LBFGS,
    "rangeradabelief": RangerAdaBelief,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD,
    "sparseadam": torch.optim.SparseAdam,
}

from .base import Optimizer

from .lookahead import *
from .ralamb import *
from .rangerlars import *


def get_optimizer(s):
    s = s.lower()
    return optimizer_lookup[s]


get = get_optimizer

from .schedulers import get_scheduler
