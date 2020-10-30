"""

PyTorch Hashing code is based on code from:
    https://github.com/ma3oun/hrn

Values are stored as gradient-less parameters so they get properly saved.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def _inf_normalize(x):
    """ Normalize tensor by the infinite norm.

    Just fancy talk for dividing by the maximum magniutde.
    """

    return F.normalize(torch.flatten(x, 1), p=float("inf"))


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
            prime = list_of_primes[i]
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

        assert isinstance(m, int)
        self.dim_int = m
        self.dim = nn.Parameter(torch.Tensor([m]), False)

    def hash(self, x):
        # Called by ``self.forward``
        raise NotImplementedError

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Tensor of any shape; this hash function will be applied element-wise.
        """

        return self.hash(x)


class MHash(Hash):
    """ Multiplicative Universal Hashing

    First described by Lawrence Carter and Mark Wegman 
    Universal Hash Function

    See:
        https://jeffe.cs.illinois.edu/teaching/datastructures/notes/12-hashing.pdf
    """

    def __init__(self, m, p=None, a=None, b=None):
        """

        output = ((a * input + b) % p) % m

        Parameters
        ----------
        m : int
            Size of output hash.
        p : int
            Large prime number, larger than the size of the universe input
            set. Defaults to a random prime at least 10x bigger than ``m``.
        a : int
            Salt A. If not explicitly set, randomly initialized.
        b : int
            Salt B. If not explicitly set, randomly initialized.
        """

        super().__init__(m)

        if p is None:
            # Generate a random large ``p``
            offset = np.random.randint(1, 5761454)
            p = primes_index(offset)

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
            assert 0 < b < p
            self.b = nn.Parameter(torch.LongTensor([b]), False)
        else:
            self.b = nn.Parameter(torch.randint(1, p, (1,)), False)

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

    def __repr__(self):
        elems = [self.__class__.__name__, '(']
        for i, (name, param) in enumerate(self.named_parameters()):
            if i > 0:
                elems.append(', ')
            elems.append(f"{name}={int(param)}")
        elems.append(')')
        return "".join(elems)

    def hash(self, x):
        # This may overflow, but it's (probably?) not a big deal.
        return ((self.a * x + self.b) % self.p) % self.m

class BinaryMHash(MHash):
    """ Special case of MHash where the output is in the set {-1, 1}
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        p : int
            Large prime number, larger than the size of the universe input
            set. The default value is a random large prime number that should be sufficient for most use-cases.
        """
        super().__init__(2, *args, **kwargs)

    def hash(self, x):
        output = super().hash(x)  # in set {0, 1}
        output[output == 0] = -1  # in set {-1, 1}
        return output


class MHashProj(nn.ParameterDict):
    """ Hashes and projects and arbitrary-feature-length input into a 
    fixed-feature-length output.
    """

    # Relatively arbitrary number, just each newly created ``hash_phi``
    # needs a different offset.
    prime_offset = 2000

    def __init__(self, out_feat):
        """ Applies a random feature hashing function.

        This is the function PHI described in section 2 of:
            https://arxiv.org/abs/2010.05880

        Parameters
        ----------
        out_features : int
            The output hash embedding size
        """

        super().__init__()
        out_feat = int(out_feat)

        self.hash_phi = MHash.from_offset(out_feat, HashProj.prime_offset)
        HashProj.prime_offset += 1
        self.hash_xi = BinaryHash()
        self.dim = out_feat

        self._common_init()

    @classmethod
    def from_hashers(cls, hash_h, hash_xi):
        """ More advanced initialization from externally defined hashers.

        Parameters
        ----------
        hash_h : pugh_torch.modules.Hash
            Hashing function that outputs in set ``{0, 1, ..., out_feat-1}``
        hash_xi : pugh_torch.modules.Hash
            Binary hashing function that outputs in set ``{-1, 1}``
        """

        self = cls.__new__(cls)
        super(HashProj, self).__init__()

        self.hash_h = hash_h
        self.hash_xi = hash_xi
        self.dim = int(self.hash_h.dim)

        self._common_init()

        return self

    def _common_init(self):
        # Called at the end of all constructors
        self.device = 'cpu'

    def __getitem__(self, key):
        assert isinstance(key, int)

        try:
            res = super().__getitem__(str(key))
        except KeyError:
            # compute it
            res = self._compute_hash_projection(key)
            self[str(key)] = res

        return res

    def _compute_hash_projection(self, n_input_feat):
        """ Computes in_feat x out_feat projection matrix to compute the hash.

        This matrix is not trainable, and contains elements in the set:
            {-1, 0, 1}

        Returns
        -------
        torch.Tensor
        """

        jj, ii = torch.meshgrid(torch.arange(n_input_feat), torch.arange(self.dim))
        hashed_h_jj = self.hash_h(jj)
        hashed_xi_jj = self.hash_xi(jj)
        proj = (hashed_h_jj == ii) * hashed_xi_jj
        proj = proj.to(self.device)
        return nn.Parameter(proj, False)

    def to(self, *args, **kwargs):
        """ Records the device to ``self.device``
        """

        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        self.device = device
        return super().to(*args, **kwargs)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            (b, input_feat) Tensor to hash

        Returns
        -------
        torch.Tensor
            (b, output_feat) Hashed tensor
        """

        assert x.dim == 2
        n_input_feat = x.shape[-1]
        output = torch.matmul(x, self[n_input_feat])
        return output

class RandHashProj(nn.Module):
    """ If a maximum input feature length is known, then we can just intiialize
    a single projection matrix from random numbers instead of going through
    the hassle of hash functions.
    """

    def __init__(self, out_feat, in_feat_max=8192):
        """
        Parameters
        ----------
        out_feat : int
            Output feature size
        in_feat_max : int
            Maximum input feature size.
        """

        super().__init__()

        # Build the projection matrix
        jj, ii = torch.meshgrid(torch.arange(in_feat_max), torch.arange(out_feat))

        selector = torch.randint(0, out_feat, size=(in_feat_max, 1))
        selector = selector.expand(in_feat_max, out_feat)
        binary = torch.randint(0, 2, size=(in_feat_max, 1))
        binary[binary==0] = -1
        binary = binary.expand(in_feat_max, out_feat)

        selector_mask = selector == ii
        proj = selector_mask * binary
        proj = proj.type(torch.float)
        self.proj = nn.Parameter(proj, False)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, N) feature vector
        """
        # TODO: crop the proj matrix
        _, n = x.shape
        assert n <= self.proj.shape[0], f"Input tensor of shape {int(n)} dimension exceeds maximum hashing input dimension of {int(self.proj.shape[1])}"
        output = torch.matmul(x, self.proj[:n])
        return output

