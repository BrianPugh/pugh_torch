import pytest
from pugh_torch.datasets import Dataset, ROOT_DATASET_PATH


class DummyDataset(Dataset):
    pass


@pytest.fixture
def dummy():
    return DummyDataset()


def test_path(dummy):
    assert dummy.path == (ROOT_DATASET_PATH / "unknown" / "DummyDataset")
