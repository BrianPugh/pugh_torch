import pytest
import numpy as np
import torch
from pugh_torch.linalg import batch_lstsq

def test_batch_lstsq():
    n_batch = 3
    m = 10
    n = 3
    k = 2

    A_batch = torch.rand(n_batch, m, n)
    B_batch = torch.rand(n_batch, m, k)

    solution = batch_lstsq(B_batch, A_batch)

    for i in range(len(solution)):    
        X, _ = torch.lstsq(B_batch[i], A_batch[i])
        X = X[:A_batch[i].shape[1]]
        
        assert torch.allclose(solution[i], X[:A_batch.shape[1]])
