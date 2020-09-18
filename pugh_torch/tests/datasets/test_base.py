import pytest
from pugh_torch.datasets import Dataset


class DummyDataset(Dataset):
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def dummy(mocker, tmp_path):
    mocker.patch("pugh_torch.datasets.base.ROOT_DATASET_PATH", tmp_path)
    return DummyDataset()


def test_path(dummy, tmp_path):
    assert dummy.path == (tmp_path / "datasets" / "DummyDataset")


def test_downloaded_file(dummy, tmp_path):
    assert dummy.downloaded_file == (
        tmp_path / "datasets" / "DummyDataset" / "downloaded"
    )


def test_download_dataset_if_not_downloaded(mocker, dummy, tmp_path):
    mock_download = mocker.patch.object(dummy, "download")

    assert not dummy.downloaded
    dummy._download_dataset_if_not_downloaded()

    mock_download.assert_called_once()
    assert dummy.downloaded


def test_unpack_dataset_if_not_unpacked(mocker, dummy, tmp_path):
    mock_unpack = mocker.patch.object(dummy, "unpack")

    assert not dummy.unpacked
    dummy._unpack_dataset_if_not_unpacked()

    mock_unpack.assert_called_once()
    assert dummy.unpacked
