import pytest
from pugh_torch.datasets import NYUv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


""" The tests with the ``dataset`` mark need data to be available.
"""


@pytest.fixture
def train():
    return NYUv2("train")


@pytest.fixture
def val():
    return NYUv2("val")


def assert_nyuv2(loader):
    """"""
    bar = tqdm(loader)
    for i, (rgb, depth) in enumerate(bar):
        assert rgb.shape == (16, 3, 480, 640)
        assert depth.shape == (16, 480, 640)


@pytest.mark.dataset
def test_val_get(val):
    loader = DataLoader(val, batch_size=16, drop_last=True, shuffle=True)
    assert_nyuv2(loader)


@pytest.mark.dataset
def test_train_get(train):
    loader = DataLoader(train, batch_size=16, drop_last=True, shuffle=True)
    assert_nyuv2(loader)
