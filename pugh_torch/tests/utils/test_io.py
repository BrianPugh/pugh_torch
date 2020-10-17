import pytest
import pugh_torch as pt
from pugh_torch.utils.io import gdrive_download
import gdown
from pathlib import Path

GDRIVE_TEST_FILE_URL = "https://drive.google.com/file/d/1T8TuY-8w8XVsbOI15mJZQ9OKexBqUwPZ/view?usp=sharing"

def test_gdrive_download(tmp_path):
    """ Just tests that gdown by itself works at downloading our test file.
    """
    local_path = gdrive_download(GDRIVE_TEST_FILE_URL, tmp_path)

    assert local_path.read_text() == "This is a test file to testing downloading capabilties in unit/integration tests."

