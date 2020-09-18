#!/usr/bin/env python
# content of conftest.py

import pytest
from pathlib import Path
from distutils import dir_util


def pytest_addoption(parser):
    parser.addoption(
        "--visual",
        action="store_true",
        default=False,
        help="run interactive visual tests",
    )
    parser.addoption(
        "--dataset",
        action="store_true",
        default=False,
        help="run dataset downloading tests. WARNING may use considerable bandwith and take a long time. DOES NOT SAVE DATA.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--visual"):
        # --visual given in cli: do not skip visual tests
        return
    skip_visual = pytest.mark.skip(reason="need --visual option to run")
    for item in items:
        if "visual" in item.keywords:
            item.add_marker(skip_visual)

    if config.getoption("--dataset"):
        # --dataset given in cli: do not skip dataset tests
        return
    skip_dataset = pytest.mark.skip(reason="need --dataset option to run")
    for item in items:
        if "dataset" in item.keywords:
            item.add_marker(skip_dataset)


@pytest.fixture
def chelsea():
    """What a cute RGB kitty!

    Only use for visual confirmation to help debugging. Mark these tests with:
        @pytest.mark.visual

    Returns
    -------
    numpy.ndarray
        (H,W,3) uint8_t RGB test image.
    """
    from skimage import data

    return data.chelsea()


@pytest.fixture
def data_path(tmp_path, request):
    """
    Fixture responsible for searching a folder with the same name of test
    module and, if available, copying all contents to a temporary directory so
    tests can use them freely.
    """

    filename = Path(request.module.__file__)
    test_dir = filename.parent / filename.stem
    if test_dir.is_dir():
        dir_util.copy_tree(test_dir, str(tmp_path))

    return tmp_path
