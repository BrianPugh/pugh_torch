import torch

optimizer_lookup = {
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sparseadam": torch.optim.SparseAdam,
    "asgd": torch.optim.ASGD,
    "lbfgs": torch.optim.LBFGS,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD,
}

from .base import Optimizer

from .lookahead import *
from .ralamb import *
from .rangerlars import *


def get_optimizer(s):
    s = s.lower()
    return optimizer_lookup[s]


get = get_optimizer
