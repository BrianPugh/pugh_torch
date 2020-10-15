import torch
from torch import nn
import torch.nn.functional as F

class BatchLinear(nn.Linear):
    """ Linear layer that can take batched weights and bias at runtime.
    
    Technically a little wasteful because we might be allocating
    some parameters that aren't used, but its usually a very small amount of
    memory.
    """

    def forward(self, x, weight=None, bias=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, *, feat_in) Some input tensor
        weight : torch.Tensor
            (B feat_out, feat_in)  If provided, doesn't use internal weights
        bias : 
            (B, feat_out) If provided, doesn't use internal bias
        """

        if weight is None or bias is None:
            assert weight is None and bias is None
            return super().forward(x)

        return x.matmul(weight.transpose(-1, -2)) + bias.unsqueeze(-2)

