#!/usr/bin/env python
# content of conftest.py

pytest_plugins = ["helpers_namespace"]

import pytest
from pathlib import Path
import numpy as np
from distutils import dir_util
import cv2
from PIL import Image


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
    parser.addoption(
        "--network",
        action="store_true",
        default=False,
        help="Trains a toy network to real-life integration-test some code.",
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

    if config.getoption("--network"):
        # --network given in cli: do not skip network tests
        return
    skip_network = pytest.mark.skip(reason="need --network option to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


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


@pytest.helpers.register
def assert_img_equal(img1, img2, thresh=0.001, resize=True):
    """Assert two images are similar.

    Parameters
    ----------
    img1 : numpy.ndarray or PIL.Image.Image or str-like
        First image to compare. If a numpy array, assumes RGB order.
    img2 : numpy.ndarray or PIL.Image.Image or str-like
        Second image to compare. If a numpy array, assumes RGB order.
    thresh : float
        Maximum average per-pixel L2 distance.
        Defaults to 0.001.
    resize : bool
        Bilinearly resize img2 to match the dimensions of img1.
        Defaults to True.
    """

    def standardize_args(img):
        """ Transform some img representation into a numpy array """
        if isinstance(img, np.ndarray):
            pass
        elif isinstance(img, Image.Image):
            img = np.array(img)
        else:
            # Assume its something path/str-like
            img = cv2.imread(str(img))
            img[..., :3] = img[..., :3][..., ::-1]
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = img[..., None]
        return img

    img1 = standardize_args(img1)
    img2 = standardize_args(img2)

    if resize and img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    avg_diff = np.linalg.norm(img1 - img2, axis=-1).mean()

    assert avg_diff < thresh


@pytest.fixture
def assert_img_equal(request):
    """Compares the provided file to the one recorded in the tests's data_path.

    The input image has the same constraints/requirements as described in
    the helper function ``assert_img_equal``

    Usage:
            # from pugh_torch/tests/foo/bar.py
            def test_mytest(assert_img_equal):
                some_created_img = some_function()
                # Check if this img is very close to the stored one:
                assert_img_equal(some_created_img)
                # This ALWAYS saves to "pugh_torch/tests/foo/bar/mytest_0_actual.png"
                # This compares to "pugh_torch/tests/foo/bar/mytest_0.png", if available.
                # This is so you can easily rename the "actual" image after human-verification
    """

    testname = request.node.name
    filename = Path(request.module.__file__)
    test_dir = filename.parent / filename.stem
    test_dir.mkdir(exist_ok=True)

    def _img_equal(img, index=0):
        expected_file = test_dir / f"{testname}_{index}.png"
        actual_file = test_dir / f"{testname}_{index}_actual.png"
        if img.ndim == 2:
            cv2.imwrite(str(actual_file), img)
        else:
            img_bgr = img.copy()
            img_bgr[..., :3] = img_bgr[..., :3][..., ::-1]
            cv2.imwrite(str(actual_file), img_bgr)  # img is RGB, imwrite expects BGR

        if not expected_file.exists():
            raise AssertionError(
                f"{expected_file} does not exist! Check newly produced img with a command like:\n\n    feh {actual_file}\n\n"
            )

        try:
            pytest.helpers.assert_img_equal(expected_file, img)
        except Exception as e:
            raise AssertionError(f"{expected_file} differs from {actual_file}") from e

    return _img_equal
