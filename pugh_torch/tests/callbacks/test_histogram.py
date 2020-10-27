import pytest

pytest.importorskip("pytorch_lightning")

import torch
from pugh_torch.utils import TensorBoardLogger
from pugh_torch.callbacks import Histogram
from torchvision import models


@pytest.fixture
def vgg16():
    return models.vgg16()


@pytest.fixture
def fake_trainer(mocker):
    trainer = mocker.MagicMock()
    trainer.logger = mocker.MagicMock(TensorBoardLogger)
    trainer.global_step = 5555
    return trainer


def test_histogram_basic(fake_trainer, vgg16):
    callback = Histogram()

    fake_trainer.global_step = callback.logging_batch_interval
    dataloader_idx = 0

    callback.on_train_batch_end(fake_trainer, vgg16, [], None, 0, dataloader_idx)

    # This is just to confirm it was invoked
    assert len(fake_trainer.logger.experiment.add_histogram.call_args_list) == 32

    # Spot check the first one
    args, kwargs = fake_trainer.logger.experiment.add_histogram.call_args_list[0]
    assert len(args) == 2
    assert args[0] == "features/0/weight"
    assert isinstance(args[1], torch.Tensor)
    assert kwargs["global_step"] == fake_trainer.global_step
