import torch
from torch import nn
import torch.nn.functional as F
import pugh_torch as pt
from pugh_torch.modules import RandHashProj

from basis import HRNBasis

def _extract_kwargs(kwargs, *args):
    output = {}
    for arg in args:
        if arg in kwargs:
            output[arg] = kwargs[arg]
    return output

class HRNUnit(nn.Module):
    """ This is a module that computing gets routed to.

    Unit consists of 3 primary components:
        1. A set of convolution operators
        2. A basis
        3. A unique hashing function

    This class is mainly a container these three concepts.
    There shouldn't be too much actual logic in this class.
    """

    def __init__(self, cnn, feat, n_basis, **kwargs):
        """
        Parameters
        ----------
        cnn : torch.nn.Module
            A convolutional neural net.
        feat : int
            Hashing feature size
        n_basis : int
            Number of vectors to have in basis.
        kwargs
            Passed along to appropriate constructors (just basis for now).
        """

        super().__init__()

        #self.basis = HRNBasis(n_basis, **_extract_kwargs(kwargs, "aging_rate", "aging_lookback"))
        self.basis = HRNBasis(feat, n_basis, **kwargs)
        self.cnn = cnn
        self.hasher = RandHashProj(feat)

        ########################################
        # Basis Method and Property Forwarding #
        ########################################
        basis_methods = ["is_full", "is_empty", "insert_vector", "select"]
        for basis_method in basis_methods:
            if hasattr(self, basis_method):
                raise Exception(f"{self.__class__.__name__} already has attribute \"{basis_method}\" defined.")
            setattr(self, basis_method, getattr(self.basis, basis_method))

    @property
    def hash_feat(self):
        return self.hasher.out_feat

    @torch.no_grad()
    def hash(self, x):
        """ Hash an input tensor.

        Parameters
        ----------
        x : torch.Tensor
            (B, arbitrary) Feature vector

        Returns
        -------
        torch.Tensor
            (B, feat) Normalized hashed feature vector.
        """

        return self.hasher(x)

    @torch.no_grad()
    def proj(self, x):
        """ Project tensor via the basis.

        Parameters
        ----------
        x : torch.Tensor
            (B, feat) Hashed feature vector

        Returns
        -------
        torch.Tensor
            (B, feat) Basis response
        """

        return self.basis(x)


    def forward(self, x):
        """

        Returns
        -------
        y : torch.Tensor
            (B, C, H, W) Output feature Map
        h : torch.Tensor
            (B, feat) Output normalized hashed vector for the produced feature map.
        """

        y = self.cnn(x)
        h = self.hash(y.flatten(1))
        return y, h
    
