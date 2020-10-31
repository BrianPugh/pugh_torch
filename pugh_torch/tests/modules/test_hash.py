import pytest
import numpy as np
from scipy import stats 
import torch
from torch import nn
import torch.nn.functional as F
import pugh_torch as pt
import pugh_torch.modules.hash as ptmh

import matplotlib.pyplot as plt

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

@pytest.mark.skip
def test_binary_hash_default():
    """ It seems like the MHash family of hashing functions is
    fundamentally periodic, not sure if it should really be used.
    """

    # each of these integers will get mapped to either +1 or -1.
    # Its expected that there should be approximately equal +1 and -1
    n_hashers = 10000
    trials_per_hasher = 1024
    p = 0.5
    expected_mean = p * trials_per_hasher
    expected_std = np.sqrt(trials_per_hasher * p * (1-p))

    data = torch.arange(trials_per_hasher)

    n_ones = []
    n_failures = 0
    for i in range(n_hashers):
        hasher = ptmh.BinaryMHash()
        actual = hasher(data).detach().cpu().numpy()

        cum_sum = np.cumsum(actual)

        ones_mask = actual == 1
        neg_ones_mask = actual == -1

        ones_sum = ones_mask.sum()
        n_ones.append(ones_sum)
        neg_ones_sum = neg_ones_mask.sum()

        # Ensure all elemets are in set {-1, 1}
        assert ones_sum + neg_ones_sum == actual.size

        print(hasher)
        plt.plot(cum_sum);plt.show()

        # The elements should be evenly distributed. 
        # six sigma: this should fail approximately once every 294_118 times
        tol = 6 * expected_std / actual.size
        x_hat = ones_sum / actual.size
        assert 0.5 - tol < x_hat < 0.5 + tol
        #if not 0.5 - tol < x_hat < 0.5 + tol:
        #    n_failures += 1
    n_ones = np.array(n_ones)
    actual_mean = n_ones.mean()
    actual_std = n_ones.std()
    iqr = stats.iqr(n_ones, interpolation = 'midpoint')
    import ipdb as pdb; pdb.set_trace()

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

    proj = ptmh.MHashProj.from_hashers(hash_h, hash_xi)

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


def test_rand_hash_proj_basic(mocker):
    batch = 3
    in_feat = 100
    out_feat = 5
    data = torch.rand(batch, in_feat)

    proj = ptmh.RandHashProj(out_feat)
    assert proj.proj.shape == (5, 0)

    actual_10 = proj(data[:, :10])
    assert proj.proj.shape == (5, 10)

    actual_full = proj(data)
    assert proj.proj.shape == (5, in_feat)

    actual_10_2 = proj(data[:, :10])
    assert proj.proj.shape == (5, 100)

    assert torch.allclose(actual_10, actual_10_2)

def test_rand_hash_proj_state_dict():
    batch = 3
    in_feat = 100
    out_feat = 5
    data = torch.rand(batch, in_feat)

    torch.manual_seed(0)
    proj = ptmh.RandHashProj(out_feat, sparse=False)
    res_dense = proj(data)

    state_dict = proj.state_dict()

    assert state_dict['proj'].shape == (5, 100)

    # dense->dense
    new_proj = ptmh.RandHashProj(out_feat, sparse=False)
    new_proj.load_state_dict(state_dict)
    assert torch.allclose(state_dict['proj'], proj.proj)
    assert not new_proj.proj.is_sparse
    assert torch.allclose(new_proj.proj, proj.proj)

    # dense->sparse
    new_proj = ptmh.RandHashProj(out_feat, sparse=True)
    new_proj.load_state_dict(state_dict)
    assert new_proj.proj.is_sparse
    assert not state_dict['proj'].is_sparse
    assert torch.allclose(new_proj.proj.to_dense(), proj.proj)

    # Create the sparse state_dict
    state_dict = new_proj.state_dict()
    assert state_dict['proj'].is_sparse

    # sparse->dense
    new_proj = ptmh.RandHashProj(out_feat, sparse=False)
    new_proj.load_state_dict(state_dict)
    assert not new_proj.proj.is_sparse
    assert state_dict['proj'].is_sparse
    assert torch.allclose(new_proj.proj, proj.proj)

    # sparse->sparse
    new_proj = ptmh.RandHashProj(out_feat, sparse=True)
    new_proj.load_state_dict(state_dict)
    assert new_proj.proj.is_sparse
    assert state_dict['proj'].is_sparse
    assert torch.allclose(new_proj.proj.to_dense(), proj.proj)

def test_rand_hash_proj_sparsity_sanity_check():
    batch = 3
    in_feat = 11
    out_feat = 5
    data = torch.rand(batch, in_feat)

    for _ in range(10):
        hasher = ptmh.RandHashProj(out_feat, sparse=False)

        # Do a forward pass just to populate the projection_matrix
        hasher(data)

        proj = hasher.proj.detach().cpu().numpy()
        nnz = (proj != 0).sum()

        perc_nnz = nnz / proj.size

        # since each row should contain one element, the expected value is independent
        # of the size of in_feat.
        perc_expected = 1 / out_feat
        assert perc_expected == perc_nnz

def test_rand_hash_proj_dense_vs_sparse():
    batch = 3
    in_feat = 100
    out_feat = 5
    data = torch.rand(batch, in_feat)

    torch.manual_seed(0)
    proj = ptmh.RandHashProj(out_feat, sparse=False)
    res_dense = proj(data)

    torch.manual_seed(0)
    proj = ptmh.RandHashProj(out_feat, sparse=True)
    res_sparse = proj(data)

    assert torch.allclose(res_dense, res_sparse)
