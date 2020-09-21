import pytest
import numpy as np
import torch
from pugh_torch.utils.tensorboard import SummaryWriter


@pytest.fixture
def mock_add_image(mocker):
    return mocker.patch.object(SummaryWriter, "add_image")


@pytest.fixture
def mock_add_text_under_img(mocker):
    return mocker.patch(
        "pugh_torch.utils.tensorboard.add_text_under_img",
        side_effect=lambda img, label: f"annotated img placeholder {label}",
    )


class NoInitSummaryWriter(SummaryWriter):
    """For testing isolated methods"""

    def __init__(self, *args, **kwargs):
        pass


def test_parse_rgb_transform(mocker):
    mock_imagenet = mocker.patch("pugh_torch.utils.tensorboard.imagenet")
    mock_imagenet.Unnormalize.return_value = "foo"

    writer = NoInitSummaryWriter()

    assert "bar" == writer._parse_rgb_transform(None)("bar")  # identity
    mock_imagenet.Unnormalize.assert_not_called()

    assert "foo" == writer._parse_rgb_transform("imagenet")
    mock_imagenet.Unnormalize.assert_called_once()

    with pytest.raises(NotImplementedError):
        writer._parse_rgb_transform("foobar")


def test_add_ss(mock_add_image):
    """Tests different sizes of inputs and for general common operation."""

    writer = NoInitSummaryWriter()

    rgbs = torch.rand(10, 3, 480, 640)
    preds = torch.rand(10, 13, 120, 160)
    targets = torch.randint(0, 13, size=(10, 240, 320))

    writer.add_ss("foo", rgbs, preds, targets)

    mock_add_image.assert_called_once()
    args, kwargs = mock_add_image.call_args_list[0]
    actual_tag, actual_montage = args

    assert actual_tag == "foo/0"

    assert actual_montage.shape == (480, 3 * 640, 3)
    # TODO: more strict tests on the produced montage


def test_add_ss_single_label(mock_add_text_under_img, mock_add_image):
    writer = NoInitSummaryWriter()

    rgbs = torch.rand(10, 3, 480, 640)
    preds = torch.rand(10, 13, 120, 160)
    targets = torch.randint(0, 13, size=(10, 240, 320))

    writer.add_ss("foo", rgbs, preds, targets, labels="test label")

    mock_add_image.assert_called_once()
    args, kwargs = mock_add_image.call_args_list[0]
    actual_tag, actual_rgb = args
    assert actual_rgb == "annotated img placeholder test label"


def test_add_ss_multi_label(mock_add_text_under_img, mock_add_image):
    writer = NoInitSummaryWriter()

    rgbs = torch.rand(10, 3, 480, 640)
    preds = torch.rand(10, 13, 120, 160)
    targets = torch.randint(0, 13, size=(10, 240, 320))

    writer.add_ss(
        "foo", rgbs, preds, targets, labels=["test label 1", "test label 2"], n_images=2
    )

    assert len(mock_add_image.call_args_list) == 2

    # First Call
    args, kwargs = mock_add_image.call_args_list[0]
    actual_tag, actual_rgb = args
    assert actual_rgb == "annotated img placeholder test label 1"

    # Second Call
    args, kwargs = mock_add_image.call_args_list[1]
    actual_tag, actual_rgb = args
    assert actual_rgb == "annotated img placeholder test label 2"


def test_add_rgb(mock_add_image):
    writer = NoInitSummaryWriter()

    rgbs = torch.rand(10, 3, 480, 640)

    writer.add_rgb("foo", rgbs)

    mock_add_image.assert_called_once()
    args, kwargs = mock_add_image.call_args_list[0]
    actual_tag, actual_rgb = args

    assert actual_tag == "foo/0"

    assert actual_rgb.dtype == np.uint8
    assert actual_rgb.shape == (480, 640, 3)
    # TODO: more strict tests on the produced image


def test_add_rgb_single_label(mock_add_text_under_img, mock_add_image):
    writer = NoInitSummaryWriter()

    rgbs = torch.rand(10, 3, 480, 640)

    writer.add_rgb("foo", rgbs, labels="test label")

    mock_add_image.assert_called_once()
    args, kwargs = mock_add_image.call_args_list[0]
    actual_tag, actual_rgb = args
    assert actual_rgb == "annotated img placeholder test label"


def test_add_rgb_multi_label(mock_add_text_under_img, mock_add_image):
    writer = NoInitSummaryWriter()

    rgbs = torch.rand(10, 3, 480, 640)

    writer.add_rgb("foo", rgbs, labels=["test label 1", "test label 2"], n_images=2)

    assert len(mock_add_image.call_args_list) == 2

    # First Call
    args, kwargs = mock_add_image.call_args_list[0]
    actual_tag, actual_rgb = args
    assert actual_rgb == "annotated img placeholder test label 1"

    # Second Call
    args, kwargs = mock_add_image.call_args_list[1]
    actual_tag, actual_rgb = args
    assert actual_rgb == "annotated img placeholder test label 2"
