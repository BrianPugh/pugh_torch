import pytest
import torch
from torch import nn
import pugh_torch as pt


@pytest.fixture
def mock_log(mocker):
    mock = mocker.patch("pugh_torch.modules.lightning_module.log")
    return mock


class SimpleModel1(pt.LightningModule):
    def __init__(self, foo):
        super().__init__()

        self.foo = foo
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)


class SimpleModel2(pt.LightningModule):
    def __init__(self, foo):
        super().__init__()

        self.foo = foo
        self.model1 = SimpleModel1("baz")
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)


def test_weight_loading_check(mock_log):

    model1 = SimpleModel1("bar")
    model2 = SimpleModel2("bar")

    model1_weights = model1.state_dict()
    model2.load_state_dict(model1_weights, strict=False)

    expected_log = """
Weights Loaded
--------------
    fc1.weight
    fc1.bias

Weights NOT Loaded (or loaded values were identical to init)
------------------------------------------------------------
    model1.fc1.weight
    model1.fc1.bias
    model1.fc2.weight
    model1.fc2.bias
    fc2.weight
    fc2.bias

Weights NOT Loaded Due to Shape Mismatch
----------------------------------------
    fc2.weight
    fc2.bias

"""

    mock_log.info.assert_called_with(expected_log)

    # Make sure the weights were actually loaded
    model2_weights = model1.state_dict()
    assert torch.equal(model1_weights["fc1.weight"], model2_weights["fc1.weight"])
    assert torch.equal(model1_weights["fc1.bias"], model2_weights["fc1.bias"])
