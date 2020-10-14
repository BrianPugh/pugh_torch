import pytest
import pugh_torch as pt


def test_resnet50_construction():
    model = pt.models.resnet50()
