import pytest
from pugh_torch.datasets.classification import ImageNet


@pytest.fixture
def imagenet(tmp_path):
    mocker.patch("pugh_torch.datasets.base.ROOT_DATASET_PATH", tmp_path)
    return ImageNet()


@pytest.mark.dataset
def test_download_from_google_drive(imagenet):
    # TODO
    pass
