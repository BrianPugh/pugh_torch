import pytest
from pugh_torch.datasets.segmentation import ADE20k
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
            A.SmallestMaxSize(args.base_size),
            A.CropNonEmptyMaskIfExists(
                args.crop_size, args.crop_size, ignore_values=[-2, -1]
            ),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    return ADE20k("train", transform=transform)


@pytest.fixture
def val():
    transform = A.Compose(
        [
            A.SmallestMaxSize(args.base_size),
            A.CropNonEmptyMaskIfExists(
                args.crop_size, args.crop_size, ignore_values=[-2, -1]
            ),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    return ADE20k("val", transform=transform)


def assert_imagenet(loader):
    """"""
    bar = tqdm(loader)
    for i, (image, label) in enumerate(bar):
        assert image.max() <= 1
        assert image.min() >= 0
        assert image.shape == (16, 3, 224, 224)
        assert label.shape == (16,)
        # TODO: We could/should make more assertions like label range and stuff.


@pytest.mark.dataset
def test_val_get(val):
    loader = DataLoader(val, batch_size=16, drop_last=True, shuffle=True)
    assert_imagenet(loader)


@pytest.mark.dataset
def test_train_get(train):
    loader = DataLoader(train, batch_size=16, drop_last=True, shuffle=True)
    assert_imagenet(loader)
