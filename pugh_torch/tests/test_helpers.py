import pytest

from pugh_torch import helpers


def test_camel_to_snake():
    assert "some_fake_class" == helpers.camel_to_snake('SomeFakeClass')
    assert "ade20k" == helpers.camel_to_snake('ADE20k')

