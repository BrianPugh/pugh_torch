import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtest import assert_vars_change
import numpy as np

import model
from model import SIREN, FastSIREN


@pytest.fixture
def rand_inputs():
    return torch.randn(100, 2)


@pytest.fixture
def rand_targets():
    return torch.randn(
        100,
        3,
    )


def test_variables_change_siren(rand_inputs, rand_targets):
    """Basic test just to make sure the graident is making it to all parts
    of the model.
    """

    batch = [rand_inputs, rand_targets]
    model = SIREN()

    # print("Our list of parameters", [np[0] for np in model.named_parameters()])

    # do they change after a training step?
    #  let's run a train step and see
    assert_vars_change(
        model=model,
        loss_fn=F.mse_loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device="cuda:0",
    )


def test_variables_change_fast_siren(rand_inputs, rand_targets):
    """Basic test just to make sure the graident is making it to all parts
    of the model.
    """

    coords = torch.randn(3, 100, 2)
    rgb_vals = torch.randn(3, 100, 3)
    imgs = torch.randn(3, 3, 224, 224)

    batch = (coords, imgs), rgb_vals

    model = FastSIREN()

    # print("Our list of parameters", [np[0] for np in model.named_parameters()])

    # do they change after a training step?
    #  let's run a train step and see
    assert_vars_change(
        model=model,
        loss_fn=F.mse_loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device="cuda:0",
    )

