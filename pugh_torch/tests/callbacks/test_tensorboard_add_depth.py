import pytest

pytest.importorskip("pytorch_lightning")

from pugh_torch.callbacks import TensorBoardAddDepth
from pugh_torch.utils import TensorBoardLogger
import torch


@pytest.fixture
def fake_batch():
    x = torch.rand(5, 3, 224, 224)
    y = torch.rand(5, 224, 224)
    return x, y


@pytest.fixture
def fake_pl_module(mocker):
    pl_module = mocker.MagicMock()
    pl_module.last_logits = torch.rand(5, 224, 224)
    return pl_module


@pytest.fixture
def fake_trainer(mocker):
    trainer = mocker.MagicMock()
    trainer.logger = mocker.MagicMock(TensorBoardLogger)
    trainer.global_step = 5555
    return trainer


def test_callback_action(mocker, tmp_path, fake_trainer, fake_batch, fake_pl_module):
    callback = TensorBoardAddDepth()

    fake_trainer.global_step = callback.logging_batch_interval
    dataloader_idx = 0

    callback.on_train_batch_end(
        fake_trainer, fake_pl_module, [], fake_batch, 0, dataloader_idx
    )

    fake_trainer.logger.experiment.add_depth.assert_called_once()
