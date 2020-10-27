import pytest
import torch
import numpy as np
import pugh_torch as pt


@pytest.fixture
def random_input():
    return torch.rand(10, 11, 12, 13)


def test_batch_index_select_basic(random_input):
    dim = 1
    n_sample = 20

    expected_shape = list(random_input.shape)
    expected_shape[dim] = n_sample
    expected_shape = tuple(expected_shape)

    index = torch.randint(
        0, random_input.shape[dim], size=(random_input.shape[0], n_sample)
    )
    result = pt.batch_index_select(random_input, dim=dim, index=index)

    assert result.shape == expected_shape


def test_batch_index_simple():
    dim = 1
    n_sample = 20

    input = torch.Tensor(np.arange(3 * 4).reshape(3, 4))
    index = torch.LongTensor(
        np.array(
            [
                [2, 0],
                [1, 1],
                [0, 3],
            ]
        )
    )

    expected = np.array(
        [
            [2, 0],
            [5, 5],
            [8, 11],
        ],
    )

    result = pt.batch_index_select(input, dim=dim, index=index)
    result = result.detach().cpu().numpy()

    assert result.shape == (3, 2)
    assert (result == expected).all()


def test_batch_index_select_negative_index(random_input):
    dim = 3
    n_sample = 20

    expected_shape = list(random_input.shape)
    expected_shape[dim] = n_sample
    expected_shape = tuple(expected_shape)

    index = torch.randint(
        0, random_input.shape[dim], size=(random_input.shape[0], n_sample)
    )
    result = pt.batch_index_select(random_input, dim=-1, index=index)

    assert result.shape == expected_shape
