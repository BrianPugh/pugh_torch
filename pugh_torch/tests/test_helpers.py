import pytest

from pugh_torch import helpers


def test_camel_to_snake():
    assert "some_fake_class" == helpers.camel_to_snake("SomeFakeClass")
    assert "ade20k" == helpers.camel_to_snake("ADE20k")

def test_add_text_under_img(chelsea, assert_img_equal):
    annotated = helpers.add_text_under_img(chelsea, "hello world")
    assert_img_equal(annotated)

