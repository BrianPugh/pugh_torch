import pytest
from pugh_torch.datasets.segmentation import ADE20K
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


""" The tests with the ``dataset`` mark need data to be available.
"""


@pytest.fixture
def train():
    transform = A.Compose(
        [
            A.SmallestMaxSize(520),
            A.CropNonEmptyMaskIfExists(
                360,
                360,
                ignore_values=[
                    0,
                ],
            ),
            A.HorizontalFlip(),
            ToTensorV2(),
        ]
    )

    return ADE20K("train", transform=transform)


@pytest.fixture
def val():
    transform = A.Compose(
        [
            A.SmallestMaxSize(520),
            A.CropNonEmptyMaskIfExists(
                360,
                360,
                ignore_values=[
                    0,
                ],
            ),
            A.HorizontalFlip(),
            ToTensorV2(),
        ]
    )

    return ADE20K("val", transform=transform)


def assert_ade20k(loader):
    """"""
    bar = tqdm(loader)
    for i, (image, label) in enumerate(bar):
        assert image.max() <= 1
        assert image.min() >= 0
        assert image.shape == (16, 3, 360, 360)

        assert label.min() >= 0
        assert label.max() <= 150
        assert label.shape == (16, 360, 360)


@pytest.mark.dataset
def test_val_get(val):
    loader = DataLoader(val, batch_size=16, drop_last=True, shuffle=True)
    assert_ade20k(loader)


@pytest.mark.dataset
def test_train_get(train):
    loader = DataLoader(train, batch_size=16, drop_last=True, shuffle=True)
    assert_ade20k(loader)
