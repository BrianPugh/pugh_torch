import torch
from torch import nn
import torch.nn.functional as F

from collections import deque

class EmptyBasisError(Exception):
    """ Forward pass of an uninitialized/empty basis attempted.
    """

class HRNBasis(nn.Module):
    """
    """

    def __init__(self, feat, n, aging_rate=10, aging_lookback=1):
        """
        Parameters
        ----------
        feat : int
            Length of each basis vector
        n : int
            Maximum number of basis vectors
        aging_rate : float
            Multiply the ``age_thresh`` by this number every time it's reached
        aging_lookback : int
            When replacing a vector, look at this many previous minimatches
            of input feature vectors when making a decision.
            Only used during training.
        """
        super().__init__()

        # Currently basis's aren't directly learned, they are simply replaced
        # periodically (see "Aging Attributes")
        # TODO: would it be beneficial to make this parameter trainable?
        self.basis = nn.Parameter(torch.zeros(feat, n), False)

        self.init = nn.Parameter(torch.zeros(n, dtype=torch.bool), False)  # Mask of which basis has been initialized
        self.lpc = nn.Parameter(torch.zeros(n,), False)  # low projection counter

        # Aging Attributes
        # Every time a basis is selected, the age gets incremented.
        # Once age_thresh is reached, the "weakest" vector is removed. And age_thresh is increased geometrically.
        # The removed vector is replaced with a new one.
        # Differing experiments may use different replacement techniques.
        self.age = nn.Parameter(torch.LongTensor([0]), False) # Number of times this basis has been selected.
        self.aging_rate = nn.Parameter(torch.Tensor([aging_rate]), False)
        self.age_thresh = nn.Parameter(torch.Tensor(5,), False)  # TODO: expose this if necessary
        self.prev_inputs = deque(maxlen=aging_lookback)

    @property
    def is_empty(self):
        """
        Returns
        -------
        bool
            ``True`` if basis is all-zero.
        """
        return not torch.any(self.init)

    @property
    def is_full(self):
        return torch.all(self.init)

    @torch.no_grad()
    def insert_vector(self, vector, index=None, normalize=False, reset_lpc=True):
        """ Insert ``vector`` into the basis.

        If the vector isn't already normalized (L2 magnitude of 1),
        you should set ``normalize=True``.

        Raises
        ------
        IndexError
            If attempting to insert a vector into a full basis set.

        Parameters
        ----------
        vector : torch.Tensor
            (feat,) vector to insert into the basis set.
        index : int
            Index of the vector slot to use.
            If not provided, defaults to the first free slot.
            Raises an ``IndexError`` if the basis is full
        normalize : bool
            L2-norm the input bector.
            Defaults to ``False``.
        reset_lpc : bool
            Set the low-projection-counter for this vector to 0.
            Defaults to ``True``.
        """

        if index is None:
            index = torch.nonzero(~self.init)[0]

        if normalize:
            vector = F.normalize(vector, dim=0)

        self.basis[:, index].copy_(vector)
        self.init[index] = True

        if reset_lpc:
            self.lpc[index] = 0

    @torch.no_grad()
    def del_vector(self, index):
        """ Zero-out a vector from the basis set.
        """

        # Zero out this basis (not strictly necessary)
        self.basis[:, index].zero_()
        # Reset the low-projection-counter for this index
        self.lpc[index] = 0
        # Mark this vector as uninitialized
        self.init[index] = False

    @torch.no_grad()
    def _update_basis(self):
        """ Replace a basis vector using information from internal-state.

        Called when age_thresh has been reached
        """

        original_mode = self.training

        # Find the lowest-projection-index:
        lpi = torch.argmin(self.lpc)

        self.del_vector(lpi)

        # Put self in eval mode so we don't update internal state during
        # the forward pass.
        self.eval()

        # Form a minibatch of the previous inputs
        input_batch = torch.cat(self.prev_inputs, dim=0)

        # Compute the average residual
        output_projection = self(input_batch)
        residual = input_batch - output_projection
        residual = residual.mean(dim=0)

        # Insert the residual into the basis
        self.insert_vector(residual, normalize=True)

        # Put the module back into the state before calling this method
        self.train(original_mode)


    @torch.no_grad()
    def select(self):
        """ Increment internal state that this basis was selected.

        This isn't exactly Algorithm 2 in the paper, but its similar
        """

        # Increment low projection counter of the basis output with the
        # smallest magnitude
        mags = torch.linalg.norm(self.prev_unreduced_output, dim=1)  # (B, n_initialized)
        lp_index = torch.argmin(mags, dim=-1)  # (B, )
        lp_index, lp_count = torch.unique(lp_index, return_counts=True)
        self.lpc[lp_index] += lp_count

        # Increment age
        self.age += 1
        if self.age >= self.age_thresh:
            # Vector Update/Replacement Algorithm
            self._update_basis()

            # Update age threshold and reset counters
            self.age_thresh *= self.aging_rate
            self.age.zero_()
            self.lpc.zero_()

    def forward(self, x):
        """ Computes the inner product

        Parameters
        ----------
        x : torch.Tensor
            (B, feat) input tensor

        Returns
        -------
        torch.Tensor
            (B, )
        """

        if self.is_empty:
            raise EmptyBasisError

        if self.training:
            # Save the last input for potential future basis updates 
            self.prev_inputs.append(x.detach())

        x = x.unsqueeze(-1)  # (B, feat, 1)

        basis = self.basis[:, self.init]  # (feat, n_initialized)
        basis = basis.unsqueeze(0)  # (1, feat, n_initialized)

        output = (x * basis)  # (B, feat, n_initialized)

        if self.training:
            self.prev_unreduced_output = output.detach()

        output = output.sum(-1)  # (B, feat)

        return output
