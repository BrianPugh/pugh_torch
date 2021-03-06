class HashMismatchError(Exception):
    """Occurs when a file does not have the expected hash."""


class DataUnavailableError(Exception):
    """Data is missing from disk and isn't readily available to easily download."""


class ShouldNeverHappenError(Exception):
    """Portions of code have been reached that the developer never intended."""
