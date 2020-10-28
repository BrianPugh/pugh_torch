import pytest
import numpy as np
import torch
from pugh_torch.linalg import batch_lstsq

_rtol = 1e-4
_atol = 1e-4


def test_batch_lstsq():
    n_batch = 3
    m = 10
    n = 3
    k = 2

    A_batch = torch.rand(n_batch, m, n)
    B_batch = torch.rand(n_batch, m, k)

    solution = batch_lstsq(B_batch, A_batch)

    assert solution.shape == (n_batch, n, k)

    for i in range(len(solution)):
        X, _ = torch.lstsq(B_batch[i], A_batch[i])
        X = X[: A_batch[i].shape[1]]

        assert torch.allclose(solution[i], X, rtol=_rtol, atol=_atol)


def test_batch_lstsq_ragged_experiment():
    """Just tests pytorch's ability to handle 0 rows"""

    n_batch = 3
    m = 10
    n = 3
    k = 2

    A_batch = torch.rand(n_batch, m - 2, n)
    A_zeros = torch.zeros(n_batch, 2, n)
    A_batch_padded = torch.cat((A_batch, A_zeros), dim=1)

    B_batch = torch.rand(n_batch, m - 2, k)
    B_zeros = torch.zeros(n_batch, 2, k)
    B_batch_padded = torch.cat((B_batch, B_zeros), dim=1)

    solution = batch_lstsq(B_batch, A_batch)
    solution_padded = batch_lstsq(B_batch_padded, A_batch_padded)

    # The 0 padding shouldn't influence the solution.
    assert torch.allclose(solution, solution_padded, rtol=_rtol, atol=_atol)
