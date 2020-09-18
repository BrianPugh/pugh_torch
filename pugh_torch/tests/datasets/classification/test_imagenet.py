import pytest
from pugh_torch.datasets.classification import ImageNet
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


""" The tests with the ``dataset`` mark need data to be available.
"""


@pytest.fixture
def imagenet_train(tmp_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # So that exemplars can be collated.
        transforms.ToTensor(),
        ]) 
    return ImageNet(split="train", transform=transform)


@pytest.fixture
def imagenet_val(tmp_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # So that exemplars can be collated.
        transforms.ToTensor(),
        ]) 
    return ImageNet(split="val", transform=transform)


def assert_imagenet(loader):
    """"""
    bar = tqdm(loader)
    for i, (image, label) in enumerate(bar):
        import ipdb as pdb

        pdb.set_trace()


@pytest.mark.dataset
def test_imagenet_val_get(imagenet_val):
    loader = DataLoader(imagenet_val, batch_size=16, drop_last=True, shuffle=True)
    assert_imagenet(loader)


@pytest.mark.dataset
def test_imagenet_train_get(imagenet_train):
    loader = DataLoader(imagenet_train, batch_size=16, drop_last=True, shuffle=True)
    assert_imagenet(loader)
