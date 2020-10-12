import torch
from . import optimizer_lookup


class Optimizer(torch.optim.Optimizer):
    """Only used to automatically register optimizers functions."""

    def __init_subclass__(cls, **kwargs):
        """Automatic registration stuff"""
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        if name in optimizer_lookup:
            raise ValueError(
                f'Optimizer "{name}" already exists: {optimizer_lookup[name]}'
            )
        optimizer_lookup[name] = cls
