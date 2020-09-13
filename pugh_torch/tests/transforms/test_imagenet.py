from pugh_torch.transforms import imagenet
import numpy as np  
import torch
import pytest


def test_normalize_numpy():
    img = np.array(
        [
            [[0.2, 0.5, 0.8]],
        ]
        )  # (1,1,3) image
    transform = imagenet.Normalize()
    actual = transform(img)

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == (3, 1, 1)
    assert np.isclose(actual[0], (0.2 -0.485) / 0.229)
    assert np.isclose(actual[1], (0.5 -0.456) / 0.224)
    assert np.isclose(actual[2], (0.8 -0.406) / 0.225)


def test_normalize_greater_1_value_error():
    
    img = np.array(
        [
            [[50, 127, 250]],
        ]
        )  # (1,1,3) image
    transform = imagenet.Normalize()
    with pytest.raises(ValueError):
        actual = transform(img)


def test_normalize_uint8_value_error():
    
    img = np.array(
        [
            [[0, 0, 0]],
        ],
        dtype=np.uint8)  # (1,1,3) image
    transform = imagenet.Normalize()
    with pytest.raises(ValueError):
        actual = transform(img)


def test_normalize_tensor():
    img = torch.FloatTensor(
        [
            [[0.2, 0.5, 0.8]],
        ]
        )  # (1, 1, 3) image
    img = img.permute(2, 0, 1)  # (3, H, W)
    transform = imagenet.Normalize()
    actual = transform(img)
    assert actual.shape == (3, 1, 1)
    assert np.isclose(actual[0], (0.2 -0.485) / 0.229)
    assert np.isclose(actual[1], (0.5 -0.456) / 0.224)
    assert np.isclose(actual[2], (0.8 -0.406) / 0.225)


def test_auto_normalize_unnormalize():
    img = np.array(
        [
            [[0.2, 0.5, 0.8]],
        ]
        )  # (1,1,3) image
    normalize = imagenet.Normalize()
    unnormalize = imagenet.Unnormalize()

    actual = unnormalize(normalize(img))
    assert isinstance(actual, torch.Tensor)
    actual = actual.numpy().transpose(1,2,0)
    assert np.isclose(img, actual).all()

