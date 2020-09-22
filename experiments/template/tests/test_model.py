import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtest import assert_vars_change

from model import MyModel as Model


@pytest.fixture
def rand_inputs():
    return torch.randn(2, 3, 100, 100)


@pytest.fixture
def rand_targets():
    return torch.randint(0, 10, (2,), dtype=torch.long)


def test_variables_change(rand_inputs, rand_targets):
    """Basic test just to make sure the graident is making it to all parts
    of the model.
    """

    batch = [rand_inputs, rand_targets]
    model = Model(num_classes=10)

    # print("Our list of parameters", [np[0] for np in model.named_parameters()])

    # do they change after a training step?
    #  let's run a train step and see
    assert_vars_change(
        model=model,
        loss_fn=F.cross_entropy,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device="cuda:0",
    )
