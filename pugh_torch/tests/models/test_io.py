import pytest
import pugh_torch as pt
import gdown
from pathlib import Path
import torch
from torch import nn

GDRIVE_TEST_FILE_URL = (
    "https://drive.google.com/file/d/1T8TuY-8w8XVsbOI15mJZQ9OKexBqUwPZ/view?usp=sharing"
)


def assert_state_dict_equal(expected, actual):
    for key in expected:
        assert (expected[key] == actual[key]).all()


def test_load_state_dict_from_url(mocker, tmp_path):
    """Tests normal downloading and loading a checkpoint from a url"""

    state_dict_path = tmp_path / "foo.pth"

    # Create a state_dict
    network = nn.Linear(2, 4)
    expected_state_dict = network.state_dict()

    # Save it to a file
    torch.save(expected_state_dict, state_dict_path)

    mock_gdrive_download = mocker.patch("pugh_torch.models.io.gdrive_download")
    mock_gdrive_download.return_value = state_dict_path

    actual_state_dict = pt.models.io.load_state_dict_from_url(
        "https://drive.google.com/fake_url", state_dict_path
    )

    assert_state_dict_equal(expected_state_dict, actual_state_dict)
