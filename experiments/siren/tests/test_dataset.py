import pytest
import torch
import numpy as np
from dataset import SingleImageDataset
from torch.utils.data import DataLoader


@pytest.fixture
def chelsea():
    from skimage import data

    return data.chelsea()


@pytest.fixture
def simple_img():
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[0, 0, :] = [1, 2, 3]
    img[-1, 0, :] = [4, 5, 6]
    img[0, -1, :] = [7, 8, 9]
    return img


@pytest.fixture
def random_img():
    img = np.random.randint(0, 256, size=(5, 5, 3))
    return img


def test_single_image_dataset_train(mocker, simple_img):
    """Tests basic training operation using nonrandom values"""

    # This is in X, Y cooridnates
    fake_random_data = torch.Tensor(
        np.array(
            [
                [0, 0],  # top left
                [0, 1],  # bottom left
                [1, 0],  # top right
            ]
        )
    )

    mock_random = mocker.patch("dataset.torch.rand")
    mock_random.return_value = fake_random_data

    dataset = SingleImageDataset(simple_img, 3, normalize=False)
    assert len(dataset) == 9

    dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)

    n_iter = 0
    for x, y in dataloader:
        n_iter += 1
        x = x.numpy()
        y = y.numpy()
        expected_x = np.array(
            [
                [-1, -1],  # Top left
                [-1, 1],  # Bottom left
                [1, -1],  # Top right
            ]
        )
        assert (expected_x == x).all()

        expected_y = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        assert (expected_y == y).all()

    assert n_iter == 9


def test_single_image_dataset_train_interpolation(mocker, simple_img):
    """Interpolate very positionally similar values"""

    # This is in X, Y cooridnates
    fake_random_data = torch.Tensor(
        np.array(
            [
                [0.001, 0],  # top left
                [0, 0.999],  # bottom left
                [0.999, 0],  # top right
            ]
        )
    )

    mock_random = mocker.patch("dataset.torch.rand")
    mock_random.return_value = fake_random_data

    dataset = SingleImageDataset(simple_img, 3, normalize=False)
    assert len(dataset) == 9

    dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)

    n_iter = 0
    for x, y in dataloader:
        n_iter += 1
        x = x.numpy()
        y = y.numpy()
        expected_x = np.array(
            [
                [-0.998, -1],  # Top left
                [-1, 0.998],  # Bottom left
                [0.998, -1],  # Top right
            ]
        )
        assert np.isclose(expected_x, x).all()

        expected_y = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        assert np.isclose(expected_y, y, atol=0.05).all()

    assert n_iter == 9


def test_single_image_dataset_val(random_img):
    dataset = SingleImageDataset(random_img, 3, mode="val", normalize=False)
    dataloader = DataLoader(dataset, batch_size=None, batch_sampler=None)

    actual_img = np.zeros_like(random_img)
    for x, y in dataloader:
        x = x.numpy()
        y = y.numpy()

        # Unnormalize x
        x += 1
        x /= 2
        x *= 4  # shape - 1
        x = x.astype(np.int)

        actual_img[x[:, 1], x[:, 0]] = y

    assert (np.isclose(random_img, actual_img)).all()
