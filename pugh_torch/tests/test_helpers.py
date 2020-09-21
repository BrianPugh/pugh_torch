import pytest
import cv2
from pugh_torch import helpers


def test_camel_to_snake():
    assert "some_fake_class" == helpers.camel_to_snake("SomeFakeClass")
    assert "ade20k" == helpers.camel_to_snake("ADE20k")


def test_add_text_under_img(chelsea, assert_img_equal):
    annotated = helpers.add_text_under_img(chelsea, "hello world")
    assert_img_equal(annotated)


def test_add_text_under_img_multiline(chelsea, assert_img_equal):
    annotated = helpers.add_text_under_img(
        chelsea, "hello world\nmultiline test\nabc123"
    )
    assert_img_equal(annotated)


def test_add_text_under_img_tiny(chelsea, assert_img_equal):
    chelsea = cv2.resize(chelsea, (48, 48))
    annotated = helpers.add_text_under_img(
        chelsea, "this string is much longer than the image is wide."
    )
    assert_img_equal(annotated)


def test_add_text_under_img_grayscale(chelsea, assert_img_equal):
    chelsea = cv2.cvtColor(chelsea, cv2.COLOR_RGB2GRAY)
    assert chelsea.ndim == 2
    annotated = helpers.add_text_under_img(chelsea, "this is a grayscale image.")
    assert_img_equal(annotated)
