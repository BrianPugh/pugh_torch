import pytest

pytest.importorskip("pytorch_lightning")

from pugh_torch.callbacks import TensorBoardAddSS
from pugh_torch.utils import TensorBoardLogger
import torch


@pytest.fixture
def fake_batch():
    x = torch.rand(5, 3, 224, 224)
    y = torch.randint(0, 3, (224, 224))
    return x, y


@pytest.fixture
def fake_pl_module(mocker):
    pl_module = mocker.MagicMock()
    pl_module.last_logits = torch.rand(5, 10)
    return pl_module


@pytest.fixture
def fake_trainer(mocker):
    trainer = mocker.MagicMock()
    trainer.logger = mocker.MagicMock(TensorBoardLogger)
    trainer.global_step = 5555
    return trainer


def test_callback_action(mocker, tmp_path, fake_trainer, fake_batch, fake_pl_module):
    callback = TensorBoardAddSS()

    fake_trainer.global_step = callback.logging_batch_interval
    dataloader_idx = 0

    callback.on_train_batch_end(
        fake_trainer, fake_pl_module, [], fake_batch, 0, dataloader_idx
    )

    fake_trainer.logger.experiment.add_ss.assert_called_once()
    args, kwargs = fake_trainer.logger.experiment.add_ss.call_args_list[0]
    assert args[0] == "train/output"
    assert (args[1] == fake_batch[0]).all()
    assert (args[2] == fake_pl_module.last_logits).all()
    assert (args[3] == fake_batch[1]).all()
    assert kwargs["global_step"] == 20


def test_callback_skip(mocker, tmp_path, fake_batch, fake_trainer):
    callback = TensorBoardAddSS()
    batch_idx = callback.logging_batch_interval - 2
    dataloader_idx = 0

    callback.on_train_batch_end(
        fake_trainer, fake_pl_module, [], fake_batch, batch_idx, dataloader_idx
    )
    fake_trainer.logger.experiment.add_ss.assert_not_called()
