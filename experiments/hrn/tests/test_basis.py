import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from basis import HRNBasis, EmptyBasisError


def test_basis_properties():
    feat = 10
    n_basis = 7

    basis = HRNBasis(feat, n_basis)
    assert basis.n == n_basis
    assert basis.feat == feat
    assert basis.is_empty == True
    assert basis.is_full == False


def test_basis_forward():
    batch = 3
    feat = 10
    n_basis = 7

    basis_vector = torch.rand(feat)

    data = torch.rand(batch, feat)

    basis = HRNBasis(feat, n_basis)

    with pytest.raises(EmptyBasisError):
        output = basis(data)

    basis.insert(basis_vector)

    output = basis(data)

    assert output.shape == (batch, feat)


def test_basis_crud_vector():
    batch = 2
    feat = 10
    n_basis = 3

    vector0 = torch.rand(feat)
    vector1 = torch.rand(feat)
    vector2 = torch.rand(feat)

    basis = HRNBasis(feat, n_basis)

    assert basis.is_empty

    # Should go in first slot
    basis.insert(vector0, normalize=False)
    assert not basis.is_empty
    assert torch.allclose(basis.basis[:, 0], vector0)
    assert basis.init[0] == True

    # Should go in second slot, leaving slot 0 unaffected
    basis.insert(vector1, normalize=False)
    assert not basis.is_empty
    assert torch.allclose(basis.basis[:, 0], vector0)
    assert torch.allclose(basis.basis[:, 1], vector1)
    assert basis.init[1] == True

    # Changing the src vectors shouldn't impact the contents (a copy should be made)
    vector0[0] = -1
    assert not torch.allclose(basis.basis[:, 0], vector0)

    # Insert based on a provided index
    basis.insert(vector0, index=0, normalize=False)
    assert torch.allclose(basis.basis[:, 0], vector0)

    # Insert a third vector, so it should be full now
    basis.insert(vector2, normalize=False)
    assert basis.is_full
    assert torch.allclose(basis.basis[:, 2], vector2)

    # Delete a vector
    basis.delete(0)
    assert not basis.is_empty
    assert not basis.is_full
    assert (basis.basis[:, 0] == 0).all()
    assert basis.init[0] == False


def test_repr():
    feat = 10
    n_basis = 3

    basis = HRNBasis(feat, n_basis)
    actual = basis.__repr__()
    assert actual == "HRNBasis(feat=10, n=3)"


@pytest.fixture
def full_basis():
    feat = 10
    n_basis = 3

    basis = HRNBasis(feat, n_basis)
    basis.insert(torch.rand(feat))
    basis.insert(torch.rand(feat))
    basis.insert(torch.rand(feat))

    return basis


def test_basis_state_dict(full_basis):
    state_dict = full_basis.state_dict()

    assert state_dict["basis"].shape == (10, 3)
    assert state_dict["init"].shape == (3,)
    assert state_dict["lpc"].shape == (3,)
    assert state_dict["age"] == 0
    assert state_dict["aging_rate"] == 10
    assert state_dict["age_thresh"] == 5

    new_basis = HRNBasis(10, 3)
    new_basis.load_state_dict(state_dict)


def test_basis_aging(mocker, full_basis):
    basis = full_basis

    batch = 2
    feat = basis.feat
    n_basis = basis.n

    basis = HRNBasis(feat, n_basis)
    _update_basis_mock = mocker.patch.object(basis, "_update_basis")
    basis.insert(torch.rand(feat))
    basis.insert(torch.rand(feat))
    basis.insert(torch.rand(feat))
    basis.train()

    assert basis.age == 0

    # Age should not increase if this basis was not selected.
    for i in range(5):
        data = torch.rand(batch, feat)
        output = basis(data)
        assert basis.age == 0
        # Make sure ``prev_input`` IS recorded
        assert torch.allclose(basis.prev_input, data)
        # Make sure ``prev_inputs`` are NOT recorded
        assert len(basis.prev_inputs) == 0

    # Age should increase when selected
    for i in range(4):
        data = torch.rand(batch, feat)
        output = basis(data)
        basis.select()
        assert basis.age == (i + 1)
        # Make sure previous inputs are recorded
        assert torch.allclose(basis.prev_inputs[-1], data)

    # The next selection should trigger the basis update mechanism
    # We'll test that method more indepth in another test.
    _update_basis_mock.assert_not_called()
    basis.select()
    _update_basis_mock.assert_called_once()

    # Make sure all the counters got reset
    assert basis.age == 0
    assert (basis.lpc == 0).all()
    assert basis.age_thresh == 50


def test_basis_update_basis(mocker):
    batch = 2
    feat = 5
    n_basis = 3

    basis = HRNBasis(feat, n_basis)

    vector0 = torch.zeros(feat)
    vector0[0] = 1
    vector1 = torch.zeros(feat)
    vector1[1] = 1
    vector2 = torch.zeros(feat)
    vector2[2] = 1

    basis.insert(vector0)
    basis.insert(vector1)
    basis.insert(vector2)

    insert_spy = mocker.spy(basis, "insert")
    delete_spy = mocker.spy(basis, "delete")

    feat = basis.feat
    n_basis = basis.n

    basis.train()
    assert basis.training == True

    # Populate a ``prev_inputs``
    data = torch.ones(batch, feat)
    basis.prev_inputs.append(data)

    # Populate a fake lpc that forces vector2 to be selected as the worst
    basis.lpc.fill_(0)
    basis.lpc[2] = 5

    basis._update_basis()

    # Make sure we're still in training mode
    assert basis.training == True

    # Verify the update
    expected_residual = torch.Tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    delete_spy.assert_called_once_with(2)
    insert_spy.assert_called_once()
    args, kwargs = insert_spy.call_args_list[0]
    assert torch.allclose(args[0], expected_residual)
