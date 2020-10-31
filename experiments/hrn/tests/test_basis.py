import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from basis import HRNBasis

def test_basis_forward():
    batch = 3
    feat = 10
    n_basis = 7

    data = torch.rand(batch, feat)

    basis = HRNBasis(feat, n_basis)

    data = basis(data)

    assert data.shape == (batch, feat)

def test_basis_insert_vector():
    batch = 3
    feat = 10
    n_basis = 7

    vector0 = torch.rand(feat)
    vector1 = torch.rand(feat)

    basis = HRNBasis(feat, n_basis)

    # Should go in first slot
    basis.insert_vector(vector0)
    assert torch.allclose(basis.basis[:, 0], vector0)

    # Should go in second slot, leaving slot 0 unaffected
    basis.insert_vector(vector1)
    assert torch.allclose(basis.basis[:, 0], vector0)
    assert torch.allclose(basis.basis[:, 1], vector1)

    # Changing the src vectors shouldn't impact the contents (a copy should be made)
    vector0[0] = -1
    assert not torch.allclose(basis.basis[:, 0], vector0)

    # Insert based on a provided index
    basis.insert_vector(vector0, index=0)
    assert torch.allclose(basis.basis[:, 0], vector0)


