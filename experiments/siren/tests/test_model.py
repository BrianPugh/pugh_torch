import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtest import assert_vars_change

from model import SIREN


@pytest.fixture
def rand_inputs():
    return torch.randn(100, 2)


@pytest.fixture
def rand_targets():
    return torch.randn(
        100,
        3,
    )


def test_variables_change(rand_inputs, rand_targets):
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
