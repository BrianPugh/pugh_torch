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
        super(MHashProj, self).__init__()

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
    """ We can just extend a single projection matrix without the
    need for two separate hash functions.

    This algorithm deterministically maps an arbitrarily long ``in_feat``
    vector into a fixed-length ``out_feat`` vector. It accomplishes this by
    the following algorithm:
        For each element in the input feature vector:
            1. Based on the index, deterministically multiply it by ``1`` or ``-1``
            2. Based on the index, deterministically map it to a single element
               in the output feature vector.
        Each element in the output feature vector is the sum of all the 
        input elements mapped to it.

    Attributes
    ----------
    proj : torch.nn.Parameter
        (out_feat, in_feat) where in_feat is the maximum input feature
        size fed through yet.
    """

    def __init__(self, out_feat, sparse=None):
        """
        Parameters
        ----------
        out_feat : int
            Output feature size
        sparse : bool
            Use a sparse representation for the internal projection matrix.
            Saves a good amount of memory when ``out_feat>5``, which is a pretty
            typical use-case.
            Defaults to whichever representation would be more memory efficient.
        """

        super().__init__()

        # Automatically set ``sparse`` depending on output feature size.
        if sparse is None:
            if out_feat > 5:
                sparse = True
            else:
                sparse = False

        # Only use the following attribute for loading from state_dict
        self.__init_sparse = sparse

        if sparse:
            self.proj = nn.Parameter(torch.sparse.FloatTensor(out_feat, 0), False)
        else:
            self.proj = nn.Parameter(torch.Tensor(out_feat, 0), False) 

    def __repr__(self):
        return f"{self.__class__.__name__}(out_feat={int(self.proj.shape[0])})"

    @property
    def sparse(self):
        return self.proj.is_sparse

    @torch.no_grad()
    def _get_proj(self, in_feat_new):
        """ Returns a view of the projection matrix

        Parameters
        ----------
        in_feat_new : int
            Number of features that need projecting

        Returns
        -------
        torch.Parameter
            (in_feat_new, out_feat) projection matrix.
        """

        out_feat, in_feat_old = self.proj.shape

        if in_feat_new <= in_feat_old:
            # We can just return a view of our currently stored projection
            # matrix
            if self.sparse:
                # Have to do some hacky stuff because strides aren't available
                # directly when using sparse tensors
                indices = self.proj.indices() # (2, N) sorted because its coalesced
                values = self.proj.values()

                mask = indices[1] < in_feat_new

                proj = torch.sparse.FloatTensor(indices[:, mask], values[mask], (out_feat, in_feat_new))
                return proj
            else:
                return self.proj[:, :in_feat_new]

        ext = self._get_dense_ext(in_feat_new)

        if self.sparse:
            ext = ext.to_sparse()

        new_proj = torch.cat((self.proj, ext), dim=1)

        if self.sparse:
            new_proj = new_proj.coalesce()

        self.proj = nn.Parameter(new_proj, False)

        return self.proj

    @torch.no_grad()
    def _get_dense_ext(self, in_feat_new):
        """ Create the new extension portion of projection matrix
        """

        # Extend the existing projection matrix
        out_feat, in_feat_old  = self.proj.shape
        in_feat_diff = in_feat_new - in_feat_old
        device = self.proj.device

        # We have to extend our existing projection matrix

        ii = torch.arange(0, out_feat, device=device)
        ii = ii.unsqueeze(-1)
        ii = ii.expand(-1, in_feat_diff)

        selector = torch.randint(0, out_feat, size=(1, in_feat_diff), device=device)
        selector = selector.expand(out_feat, -1)

        selector_mask = selector == ii

        binary = torch.randint(0, 2, size=(1, in_feat_diff), device=device)
        binary[binary==0] = -1
        binary = binary.expand(out_feat, -1)

        ext = selector_mask * binary
        ext = ext.type(self.proj.dtype)

        return ext

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """ Normally, pytorch will raise an exception because the existing
        parameter ``proj`` and the one in the state dict will mismatch along
        dimension(1), the in_feat dimension.
        This override will allow for the successful load of the projection matrix.
        """

        # Pop "proj" so that we can load it using our special rules.
        proj = state_dict.pop('proj')
        out_feat, in_feat = proj.shape
        self_out_feat = self.proj.shape[0]
        assert out_feat == self_out_feat, f"State dict out_feat={out_feat} mismatches object's out_feat={self_out_feat}"

        del self.proj  # so that upstream loading from state dict doesn't look for it

        # Perform the loading; the parent method will perform the data copy
        super()._load_from_state_dict(
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                )

        # Create the projection matrix of the same shape, but the __init__
        # sparse/dense configuration.
        if self.__init_sparse:
            self.proj = nn.Parameter(torch.sparse.FloatTensor(out_feat, in_feat), False)
            if proj.is_sparse:
                self.proj.copy_(proj)
            else:
                self.proj.copy_(proj.to_sparse())
        else:
            self.proj = nn.Parameter(torch.Tensor(out_feat, in_feat), False) 
            if proj.is_sparse:
                self.proj.copy_(proj.to_dense())
            else:
                self.proj.copy_(proj)

        # Re-add it to the dictionary so that 
        state_dict['proj'] = proj

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, N) feature vector
        """

        _, n = x.shape
        proj = self._get_proj(n)
        # Matmul only supporse (sparse, dense) multiply, not (dense, sparse)
        # So we do the matmul (and the proj dimensions) sort of "backwards"
        # from intuitive
        output = torch.matmul(proj, x.transpose(0, 1))
        output = output.transpose(0, 1)
        return output

