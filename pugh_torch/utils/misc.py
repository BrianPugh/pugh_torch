from contextlib import contextmanager
from time import time


@contextmanager
def timeit(msg=""):
    """Timing context manager

    Example usage:
        >>> with timeit('Data Reading'):
        ...    time.sleep(5)
        Data Reading executed in 5.002 seconds.

    Parameters
    ----------
    msg : str
        Message to be prepended to " executed in %.3f seconds."
    """

    try:
        print(f'Starting "{msg}" ...')
        t_start = time()
        yield
    finally:
        t_end = time()
        t_diff = t_end - t_start
        print(f"{msg} executed in {t_diff} seconds")
