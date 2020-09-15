# -*- coding: utf-8 -*-

"""Top-level package for Pugh Torch."""

__author__ = "Brian Pugh"
__email__ = "bnp117@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.2.0"


def get_module_version():
    return __version__


from .example import Example  # noqa: F401
