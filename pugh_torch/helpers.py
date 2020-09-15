import re
import hashlib


def camel_to_snake(s):
    """Converts a camelCase or pascalCase string to snake_case

    Parameters
    ----------
    s : str
        Some camel/pascal case string to convert to snake case

    Returns
    -------
    str
        camel_case equivalent of the provided string.
    """

    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", s)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def compute_sha1(path, chunk_size=2 ** 20):
    """Computes the SHA1 hash of a file

    Parameters
    ----------
    path : str-like
        Path to file to compute the hash of.
    """

    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def compare_hash(expected, actual):
    """
    Raises
    ------
    pugh_torch.HashMismatchError
        If the hashes do not match.
    """

    if expected == actual:
        return

    raise HashMismatchError(f"{actual} doesn't match expected hash {expected}")
