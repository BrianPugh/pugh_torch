import pytest
from pugh_torch.datasets.classification import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


""" The tests with the ``dataset`` mark need data to be available.
"""


@pytest.fixture
def train(tmp_path):
    return MNIST(split="train")


@pytest.fixture
def val(tmp_path):
    return MNIST(split="val")


def assert_imagenet(loader):
    """"""
    bar = tqdm(loader)
    for i, (image, label) in enumerate(bar):
        assert image.max() <= 1
        assert image.min() >= 0
        assert image.shape == (16, 1, 28, 28)
        assert label.shape == (16,)
        assert label.max() <= 9
        assert label.min() >= 0


@pytest.mark.dataset
def test_val_get(val):
    loader = DataLoader(val, batch_size=16, drop_last=True, shuffle=True)
    assert_imagenet(loader)


@pytest.mark.dataset
def test_train_get(train):
    loader = DataLoader(train, batch_size=16, drop_last=True, shuffle=True)
    assert_imagenet(loader)
