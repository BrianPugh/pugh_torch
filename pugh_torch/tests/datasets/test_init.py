import pytest

import pugh_torch as pt
from pugh_torch.datasets import DATASETS


def test_get_dataset():
    constructor = pt.datasets.get("classification", "imagenet")
    assert constructor == pt.datasets.classification.imagenet.ImageNet
