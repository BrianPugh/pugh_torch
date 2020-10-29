import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pugh_torch as pt
import pugh_torch.modules.hash as ptmh

def test_primes():
    # Exercises all code paths, doesn't validate output
    res = ptmh.primes(1000)
    res = ptmh.primes(1000)
    res = ptmh.primes(100)
    res = ptmh.primes(1000, cache=False)
    res = ptmh.primes(100, cache=False)
    res = ptmh.primes(1000, copy=True)

@pytest.fixture
def mock_torch_randint(mocker):
    """ Always returns the high value
    """
    def helper(low, high, size):
        return (high - 1) * torch.ones(size)

    mock = mocker.patch("pugh_torch.modules.hash.torch.randint")
    mock.side_effect = helper
    return mock

@pytest.fixture
def mock_np_randint(mocker):
    mock = mocker.patch("pugh_torch.modules.hash.np.random.randint")
    mock.return_value = 1000
    return mock


def test_mhash_default(mock_np_randint, mock_torch_randint):
    data = torch.ones(3, 2)
    hasher = ptmh.MHash(10)
    actual = hasher(data).detach().cpu().numpy()
    assert (actual == 1).all()

    # calls:
    #  1. ``salt a``
    #  2. ``salt_b``
    assert mock_torch_randint.call_count == 2

def test_mhash_salt(mock_np_randint, mock_torch_randint):
    data = torch.ones(3, 2)
    hasher = ptmh.MHash(10, a=1, b=1)
    actual = hasher(data).detach().cpu().numpy()
    assert (actual == 2).all()

    mock_torch_randint.assert_not_called()

def test_mhash_from_offset(mock_np_randint, mock_torch_randint):
    data = torch.ones(3, 2)
    hasher = ptmh.MHash.from_offset(10, 500)
    actual = hasher(data).detach().cpu().numpy()
    assert (actual == 1).all()

    # once for ``salt a`` and once for ``salt_b``
    assert mock_torch_randint.call_count == 2

    mock_np_randint.assert_not_called()

def test_binary_hash_default():
    #data = torch.randint(0, 100, size=(1000, 1000))
    data = torch.arange(10000)
    # each of these integers will get mapped to either +1 or -1.
    # Its expected that there should be approximately equal +1 and -1

    for i in range(100000):
        hasher = ptmh.BinaryHash()
        actual = hasher(data).detach().cpu().numpy()

        ones_mask = actual == 1
        neg_ones_mask = actual == -1

        ones_sum = ones_mask.sum()
        neg_ones_sum = neg_ones_mask.sum()

        # Ensure all elemets are in set {-1, 1}
        assert ones_sum + neg_ones_sum == actual.size

        # The elements should be evenly distributed. 
        tol = 0.05
        assert 0.5 - tol < ones_sum / actual.size < 0.5 + tol

class ModHash(ptmh.Hash):
    """ Super bad, but easy to use hash function
    """

    def hash(self, x):
        return x % self.dim_int


def test_hash_projection_computation():
    hash_h = ModHash(5)  # output feature size 5

    def hash_xi(x):
        # super bad, but easy to use xi hash
        return 1

    proj = ptmh.HashProj.from_hashers(hash_h, hash_xi)

    actual = proj[10].detach().cpu().numpy()  # input feature size 10

    expected = np.array(
            [[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.],
       [1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]],
       dtype=np.float32)

    assert (actual == expected).all()
    assert actual.shape == (10, 5)



