import pytest

pytest.importorskip("pytorch_lightning")

from pugh_torch.callbacks import TensorBoardAddClassification
from pugh_torch.utils import TensorBoardLogger
import torch


@pytest.fixture
def fake_batch():
    x = torch.rand(5, 3, 224, 224)
    torch.manual_seed(0)
    y = torch.LongTensor([4, 3, 1, 0, 2, 9])
    return x, y


@pytest.fixture
def fake_pl_module(mocker):
    pl_module = mocker.MagicMock()
    pl_module.last_logits = torch.rand(5, 10)
    return pl_module


@pytest.fixture
def classes():
    return ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


@pytest.fixture
def fake_trainer(mocker):
    trainer = mocker.MagicMock()
    trainer.logger = mocker.MagicMock(TensorBoardLogger)
    trainer.global_step = 5555
    return trainer


def test_callback_action(mocker, tmp_path, fake_trainer, fake_batch, fake_pl_module):
    callback = TensorBoardAddClassification()

    batch_idx = callback.logging_batch_interval - 1
    dataloader_idx = 0

    callback.on_train_batch_end(
        fake_trainer, fake_pl_module, fake_batch, batch_idx, dataloader_idx
    )

    fake_trainer.logger.experiment.add_rgb.assert_called_once()
    args, kwargs = fake_trainer.logger.experiment.add_rgb.call_args_list[0]
    assert args[0] == "train/output"
    assert (args[1] == fake_batch[0]).all()
    assert kwargs["global_step"] == 5555
    assert kwargs["labels"] == [
        "Truth: 4 (N/A)\nPred: 7 (N/A)",
        "Truth: 3 (N/A)\nPred: 7 (N/A)",
        "Truth: 1 (N/A)\nPred: 6 (N/A)",
        "Truth: 0 (N/A)\nPred: 1 (N/A)",
        "Truth: 2 (N/A)\nPred: 6 (N/A)",
    ]


def test_callback_skip(mocker, tmp_path, fake_batch, fake_trainer):
    callback = TensorBoardAddClassification()
    batch_idx = callback.logging_batch_interval - 2
    dataloader_idx = 0

    callback.on_train_batch_end(
        fake_trainer, fake_pl_module, fake_batch, batch_idx, dataloader_idx
    )
    fake_trainer.logger.experiment.add_rgb.assert_not_called()


def test_callback_classes(
    mocker, tmp_path, fake_trainer, fake_batch, fake_pl_module, classes
):
    callback = TensorBoardAddClassification(classes=classes)

    batch_idx = callback.logging_batch_interval - 1
    dataloader_idx = 0

    callback.on_train_batch_end(
        fake_trainer, fake_pl_module, fake_batch, batch_idx, dataloader_idx
    )

    fake_trainer.logger.experiment.add_rgb.assert_called_once()
    args, kwargs = fake_trainer.logger.experiment.add_rgb.call_args_list[0]
    assert args[0] == "train/output"
    assert (args[1] == fake_batch[0]).all()
    assert kwargs["global_step"] == 5555
    assert kwargs["labels"] == [
        f"Truth: 4 ({classes[4]})\nPred: 7 ({classes[7]})",
        f"Truth: 3 ({classes[3]})\nPred: 7 ({classes[7]})",
        f"Truth: 1 ({classes[1]})\nPred: 6 ({classes[6]})",
        f"Truth: 0 ({classes[0]})\nPred: 1 ({classes[1]})",
        f"Truth: 2 ({classes[2]})\nPred: 6 ({classes[6]})",
    ]
