import torch
from torch import nn
import torch.nn.functional as F


class HRNBasis(nn.Module):
    """
    """

    def __init__(self, feat, n):
        """
        Parameters
        ----------
        feat : int
            Length of each basis vector
        n : int
            Maximum number of basis vectors
        """
        super().__init__()

        self.basis = nn.Parameter(torch.zeros(feat, n))

        # Index along basis dimension containing a zero-vector to populate next.
        self.insert_idx = 0

    @property
    def is_empty(self):
        """
        Returns
        -------
        bool
            ``True`` if basis is all-zero.
        """
        return self.init_pos == 0

    @property
    def is_full(self):
        return self.init_pos == self.basis.shape[1]

    @torch.no_grad()
    def insert_vector(self, vector, index=None):
        """ Insert ``vector`` into the basis.

        Parameters
        ----------
        vector : torch.Tensor
            (feat,) vector to insert into the basis set.
        index : int
            Index of the vector slot to use.
            If not provided, defaults to the first free slot.
            Raises an ``IndexError`` if the basis is full
        """

        if index is None:
            self.basis[:, self.insert_idx].copy_(vector)
            self.insert_idx += 1
        else:
            self.basis[:, index].copy_(vector)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, feat) input tensor
        """

        x = x.unsqueeze(-1)  # (B, feat, 1)
        output = (x * self.basis.unsqueeze(0)).sum(-1)
        return output
