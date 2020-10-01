import pytest
import torch
from pugh_torch.losses import reduction_str


def test_reduction_str():
    assert torch.sum == reduction_str("sum")
    assert torch.sum == reduction_str("SUM")
    assert torch.mean == reduction_str("mean")

    with pytest.raises(ValueError):
        assert torch.sum == reduction_str("foo")


def test_reduction_str_none():
    op = reduction_str("none")
    assert 5 == op(5)
