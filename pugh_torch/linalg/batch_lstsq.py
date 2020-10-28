"""
Based on:
    https://github.com/pytorch/pytorch/issues/27749#issuecomment-703387840
"""

import torch

def batch_lstsq(B, A):
    """ Compute the least-squares solution.

    Finds ``x`` that minimizes ``Ax - B``

    Parameters
    ----------
    input : torch.Tensor
        (b, m, k) 
    A : torch.Tensor
        (b, m, n)

    Returns
    -------
    torch.Tensor
        (b, n, k) least squares solution.
    """
   
    X = torch.bmm(torch.pinverse(A), B)
 
    return X
