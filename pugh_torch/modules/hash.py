"""

PyTorch Hashing code isBased on code from:
    https://github.com/ma3oun/hrn

Values are stored as gradient-less parameters so they get properly saved.
"""

import torch
from torch import nn
import numpy as np


def primes(n, copy=False, cache=True):
    """ Returns a array of primes, 3 <= p < n 

    This is very fast, the following takes <1 second:
        res = primes(100_000_000)
        assert len(res) == 5_761_454
        assert res[0] == 3
        assert res[-1] == 99_999_989

    Caches the largest ``n`` array for future calls.

    Modified from:
        https://stackoverflow.com/a/3035188/13376237

    Parameters
    ----------
    n : int
        Generate primes up to this number
    copy : bool
        Copy the output array from the internal cache. Only set to ``true``
        if you intend to modify the returned array inplace.
        Defaults to ``False``.
    cache : bool
        Use the internal cache for generating/storing prime values.
        Defaults to ``True``.

    Returns
    -------
    numpy.ndarray
        Array of output primes. Do not modify this array inplace unless
        you set ``copy=True``
    """

    assert n >= 3

    if cache and n < primes.largest_n:
        index = np.searchsorted(primes.cache, n, side='right')
        output = primes.cache[:index]
    elif cache and n == primes.largest_n:
        output = primes.cache
    else:
        # This could be sped up using cache, but this is good enough
        # for now since this will usually be called with constant ``n``
        sieve = np.ones(n//2, dtype=np.bool)
        for i in range(3,int(n**0.5)+1,2):
            if sieve[i//2]:
                sieve[i*i//2::i] = False
        output = 2*np.nonzero(sieve)[0][1::]+1

        if cache:
            primes.largest_n = n
            primes.cache = output

    if copy:
        output = output.copy()

    return output
primes.largest_n = 0  # For caching primes call

def primes_index(i):
    """ Get the prime value at index.

    Parameters
    ----------
    index : int
        Index into the list of primes (starting at 3) to get.
    """

    n = 100_000_000 
    while True:
        list_of_primes = primes(n)
        try:
            prime = list_of_primes[p]
            break
        except IndexError:
            n *= 10
    return prime


class Hash(nn.Module):
    """ Base module for other pytorch hash functions.

    Attributes
    ----------
    dim : torch.Tensor
        Scalar output dimensionality (output hash size)
    """

    def __init__(self, m):
        """
        Parameters
        ----------
        m : int
            Output size of this hash function
        """

        super().__init__()
        self.dim = nn.Parameter(torch.Tensor([m]), False)

    def hash(self, x):
        # Called by ``self.forward``
        raise NotImplementedError

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of anyshape; this hash function will be applied element-wise.
        """

        return self.hash(x)


class MHash(Hash):
    """ Multiplicative Universal Hashing

    First described by Lawrence Carter and Mark Wegman 
    Universal Hash Function

    See:
        https://jeffe.cs.illinois.edu/teaching/datastructures/notes/12-hashing.pdf
    """

    def __init__(self, m, p=999_999_937, a=None, b=None):
        """

        output = ((a * input + b) % p) % m

        Parameters
        ----------
        m : int
            Size of output hash.
        p : int
            Large prime number, larger than the size of the universe input
            set. The default value is a pretty large prime number that should be sufficient for most use-cases.
        a : int
            Salt A. If not explicitly set, randomly initialized.
        b : int
            Salt B. If not explicitly set, randomly initialized.
        """

        super().__init__(m)
        assert p >= m
        self.m = nn.Parameter(torch.LongTensor([m]), False)
        self.p = nn.Parameter(torch.LongTensor([p]), False)

        # Initialize Salts
        if not a is None:
            # 0 and p lead to degenerate cases; and it's cyclic.
            assert 0 < a < p
            self.a = nn.Parameter(torch.LongTensor([a]), False)
        else:
            self.a = nn.Parameter(torch.randint(1, p, (1,)), False)

        if not b is None:
            self.b = nn.Parameter(torch.LongTensor([b]), False)
        else:
            self.b = nn.Parameter(torch.randint(1, 10000, (1,)) % self.p, False)

    @classmethod
    def from_offset(cls, m, p, *args, **kwargs):
        """ Set prime number via index into a list of primes starting from 3.

        Parameters
        ----------
        p : int
            Index into list of primes to use.
        """

        prime = primes_index(p)
        return cls(m, prime, *args, **kwargs)

    def hash(self, x):
        # This may overflow, but it's (probably?) not a big deal.
        return ((self.a * x + self.b) % self.p) % self.m

class BinaryHash(MHash):
    """ Special case of MHash where the output is in the set {-1, 1}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

    def hash(self, x):
        output = super().hash(x)  # in set {0, 1}
        output[output == 0] = -1  # in set {-1, 1}
        return output

