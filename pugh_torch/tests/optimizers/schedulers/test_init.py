import pytest
import torch
import pugh_torch as pt


def test_imports():
    assert pt.optimizers.schedulers.get_scheduler == pt.optimizers.get_scheduler
    assert pt.optimizers.get == pt.optimizers.get_optimizer
    assert pt.optimizers.schedulers.get == pt.optimizers.schedulers.get_scheduler


def test_lookups():
    def assert_lookup(cls):
        assert cls == pt.optimizers.get_scheduler(cls.__name__)

    assert_lookup(torch.torch.optim.lr_scheduler.LambdaLR)
    assert_lookup(torch.torch.optim.lr_scheduler.MultiplicativeLR)
    assert_lookup(torch.torch.optim.lr_scheduler.StepLR)
    assert_lookup(torch.torch.optim.lr_scheduler.MultiStepLR)
    assert_lookup(torch.torch.optim.lr_scheduler.ExponentialLR)
    assert_lookup(torch.torch.optim.lr_scheduler.CosineAnnealingLR)
    assert_lookup(torch.torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert_lookup(torch.torch.optim.lr_scheduler.CyclicLR)
    assert_lookup(torch.torch.optim.lr_scheduler.OneCycleLR)
    assert_lookup(torch.torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
