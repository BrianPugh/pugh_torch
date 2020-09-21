""" Tests general purpose fixtures defined in conftest.py
"""

import pytest
import numpy as np
import cv2
from PIL import Image


def test_chelsea(chelsea):
    assert isinstance(chelsea, np.ndarray)
    assert chelsea.shape == (300, 451, 3)


def test_data_path(data_path):
    fn = data_path / 'dummy_test_file'
    assert fn.is_file()


def test_assert_images_equal_numpy_grayscale():
    img1 = np.ones((5, 5))
    img2 = 0.2 * np.ones((5, 5, 1))

    pytest.helpers.assert_img_equal(img1, img1)

    with pytest.raises(AssertionError):
        pytest.helpers.assert_img_equal(img1, img2)

def test_assert_images_equal_numpy_color():
    img1 = np.ones((5, 5, 3))
    img2 = 0.2 * np.ones((5, 5, 3))

    pytest.helpers.assert_img_equal(img1, img1)

    with pytest.raises(AssertionError):
        pytest.helpers.assert_img_equal(img1, img2)

@pytest.fixture
def chelsea_file(tmp_path, chelsea):
    fn = tmp_path / 'chelsea.png'
    cv2.imwrite(str(fn), chelsea[..., ::-1])
    return fn

def test_assert_images_equal_path(chelsea_file, chelsea):
    pytest.helpers.assert_img_equal(chelsea, chelsea_file)
    pytest.helpers.assert_img_equal(chelsea_file, chelsea)

def test_assert_images_equal_str(chelsea_file, chelsea):
    chelsea_file = str(chelsea_file)
    pytest.helpers.assert_img_equal(chelsea, chelsea_file)
    pytest.helpers.assert_img_equal(chelsea_file, chelsea)

def test_assert_images_PIL(chelsea):
    chelsea_pil = Image.fromarray(chelsea)
    pytest.helpers.assert_img_equal(chelsea, chelsea_pil)
    pytest.helpers.assert_img_equal(chelsea_pil, chelsea)

def test_fixture_assert_img_equal(assert_img_equal, chelsea):
    chelsea = cv2.resize(chelsea, (100,100))
    assert_img_equal(chelsea)

def test_fixture_assert_img_equal_not_exist(mocker, assert_img_equal, chelsea):
    # The ground truth for this file doesn't exist, so this should raise
    # an assertion error.
    with pytest.raises(AssertionError):
        assert_img_equal(chelsea)
