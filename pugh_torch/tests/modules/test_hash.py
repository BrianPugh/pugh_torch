import pytest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pugh_torch.modules.hash as ptmh

def test_primes():
    # Exercises all code paths, doesn't validate output
    res = ptmh.primes(1000)
    res = ptmh.primes(1000)
    res = ptmh.primes(100)
    res = ptmh.primes(1000, cache=False)
    res = ptmh.primes(100, cache=False)
    res = ptmh.primes(1000, copy=True)


def test_mhash_blah():
    pass

def test_binary_hash():
    pass
