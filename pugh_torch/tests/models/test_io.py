import pytest
import pugh_torch as pt
import gdown
from pathlib import Path
import torch
from torch import nn
import shutil

GDRIVE_TEST_FILE_URL = (
    "https://drive.google.com/file/d/1T8TuY-8w8XVsbOI15mJZQ9OKexBqUwPZ/view?usp=sharing"
)


@pytest.fixture
def model():
    return nn.Linear(2, 4)


@pytest.fixture
def model_path(tmp_path):
    return tmp_path / "foo.pth"


@pytest.fixture
def tmp_download_path(tmp_path):
    return tmp_path / "download_cache.pth"


@pytest.fixture
def mock_gdrive_download(mocker, model, model_path, tmp_download_path):

    mock = mocker.patch("pugh_torch.models.io.gdrive_download")

    state_dict = model.state_dict()

    # Save it to a file that we'll copy over
    torch.save(state_dict, tmp_download_path)

    def side_effect(url, local):
        shutil.copy(tmp_download_path, local)
        return local

    mock.side_effect = side_effect

    return mock


def assert_state_dict_equal(expected, actual):
    for key in expected:
        assert (expected[key] == actual[key]).all()


def test_load_state_dict_from_gdrive_url(
    mocker, model, model_path, mock_gdrive_download
):
    """Tests normal downloading and loading a checkpoint from a url"""

    url = "https://drive.google.com/fake_url"
    expected_state_dict = model.state_dict()

    actual_state_dict = pt.models.io.load_state_dict_from_url(url, model_path)

    mock_gdrive_download.assert_called_once_with(url, model_path)
    assert_state_dict_equal(expected_state_dict, actual_state_dict)


def test_load_state_dict_from_gdrive_url_cached(
    mocker, model, model_path, mock_gdrive_download
):
    """Multiple calls should only result in one download."""

    url = "https://drive.google.com/fake_url"
    expected_state_dict = model.state_dict()

    actual_state_dict = pt.models.io.load_state_dict_from_url(url, model_path)
    mock_gdrive_download.assert_called_once_with(url, model_path)
    actual_state_dict = pt.models.io.load_state_dict_from_url(url, model_path)
    mock_gdrive_download.assert_called_once_with(url, model_path)

    assert_state_dict_equal(expected_state_dict, actual_state_dict)


def test_load_state_dict_from_gdrive_url_cached_force(
    mocker, model, model_path, mock_gdrive_download
):
    """Multiple calls should only result in one download."""

    url = "https://drive.google.com/fake_url"
    expected_state_dict = model.state_dict()

    actual_state_dict = pt.models.io.load_state_dict_from_url(url, model_path)
    actual_state_dict = pt.models.io.load_state_dict_from_url(
        url, model_path, force=True
    )
    mock_gdrive_download.assert_has_calls(
        [
            mocker.call(url, model_path),
            mocker.call(url, model_path),
        ]
    )

    assert_state_dict_equal(expected_state_dict, actual_state_dict)


def test_hub_url(mocker, tmp_path):
    mock = mocker.patch("pugh_torch.models.io.torch.hub.load_state_dict_from_url")
    mock.return_value = "foobar"
    state_dict = pt.models.io.load_state_dict_from_url("www.foobar.com", tmp_path)

    assert state_dict == "foobar"
    mock.asser_called_once_with(
        "www.foobar.com",
        check_hash=False,
        file_name=None,
        map_location=None,
        model_dir=None,
        progress=True,
    )
