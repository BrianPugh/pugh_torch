import pytest
from pugh_torch.utils.tensorboard import SummaryWriter


class TestSummaryWriter(SummaryWriter):
    """ For testing isolated methods
    """

    def __init__(self, *args, **kwargs):
        pass


def test_parse_rgb_transform(mocker):
    mock_imagenet = mocker.patch("pugh_torch.utils.tensorboard.imagenet")
    mock_imagenet.Unnormalize.return_value = 'foo'

    writer = TestSummaryWriter()

    assert None == writer._parse_rgb_transform(None)
    mock_imagenet.Unnormalize.assert_not_called()

    assert 'foo' == writer._parse_rgb_transform("imagenet")
    mock_imagenet.Unnormalize.assert_called_once()

    with pytest.raises(NotImplementedError):
        writer._parse_rgb_transform("foobar")

