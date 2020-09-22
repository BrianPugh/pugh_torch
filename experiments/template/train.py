import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import pugh_torch as pt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from omegaconf import DictConfig, OmegaConf
import hydra

from model import MyModel as Model

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def train(cfg):
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(1234)

    # Define albumentations transforms.
    transform = A.Compose(
        [
            pt.A.ResizeShortest(256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # this is RGB order.
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    # Instantiate your dataset
    train_dataset = pt.datasets.classification.CIFAR10("train", transform=transform)
    val_dataset = pt.datasets.classification.CIFAR10("val", transform=transform)

    # Create loaders to get data from dataset(s)
    loader_kwargs = {
        "pin_memory": True,
        "num_workers": 4,
        "batch_size": cfg.dataset.batch_size,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    # Instantiate Model to Train
    model = Model(num_classes=len(train_dataset.classes))

    # TODO add more callbacks
    trainer = Trainer(logger=pt.utils.TensorBoardLogger, **cfg.trainer)
    trainer.fit(model, train_loader, train_loader)

    # TODO: you can run the test set here, if appropriate.


if __name__ == "__main__":
    train()
