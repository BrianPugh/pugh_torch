import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from basis import HRNBasis, EmptyBasisError

def test_basis_forward():
    batch = 3
    feat = 10
    n_basis = 7

    basis_vector = torch.rand(feat)

    data = torch.rand(batch, feat)

    basis = HRNBasis(feat, n_basis)

    with pytest.raises(EmptyBasisError):
        output = basis(data)

    basis.insert_vector(basis_vector)

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
    basis.insert_vector(vector0)
    assert not basis.is_empty
    assert torch.allclose(basis.basis[:, 0], vector0)
    assert basis.init[0] == True

    # Should go in second slot, leaving slot 0 unaffected
    basis.insert_vector(vector1)
    assert not basis.is_empty
    assert torch.allclose(basis.basis[:, 0], vector0)
    assert torch.allclose(basis.basis[:, 1], vector1)
    assert basis.init[1] == True

    # Changing the src vectors shouldn't impact the contents (a copy should be made)
    vector0[0] = -1
    assert not torch.allclose(basis.basis[:, 0], vector0)

    # Insert based on a provided index
    basis.insert_vector(vector0, index=0)
    assert torch.allclose(basis.basis[:, 0], vector0)

    # Insert a third vector, so it should be full now
    basis.insert_vector(vector2)
    assert basis.is_full
    assert torch.allclose(basis.basis[:, 2], vector2)

    # Delete a vector
    basis.delete_vector(0)
    assert not basis.is_empty
    assert not basis.is_full
    assert (basis.basis[:, 0] == 0).all()
    assert basis.init[0] == False

def test_basis_state_dict():
    pass
