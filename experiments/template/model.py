""" Based on torchvision.models.resnet

This is basically resnet50, just to get you started.
"""

import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pugh_torch as pt
from pugh_torch.modules import conv3x3, conv1x1

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import albumentations as A
from albumentations.pytorch import ToTensorV2

log = logging.getLogger(__name__)

class Bottleneck(nn.Module):
    """
    Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """

    expansion = 4

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        """

        input -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> Add -> ReLU
          \_______________________(downsample if specified)_________________________/

        """

        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(out_planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, out_planes * self.expansion)
        self.bn3 = norm_layer(out_planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MyModel(pl.LightningModule):
    def __init__(
        self,
        *,
        cfg=None,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=1000,
        zero_init_residual=False,
        dilation=1,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        learning_rate=0.002,
        in_planes=64,
        **kwargs,
    ):
        """Defaults are ResNet50"""

        super().__init__()

        # NOTE: Only access this for pytorch-lightning related hooks.
        # Do not rely on cfg for network hyperparameters.
        # Use conventional arguments for constructing your architecture.
        self.cfg = cfg
        
        self.learning_rate = learning_rate # This may be overwritten by lr finder

        self.block = block = pt.to_obj(block)
        self.norm_layer = norm_layer = nn.BatchNorm2d if norm_layer is None else pt.to_obj(norm_layer)

        self.in_planes = in_planes
        self.dilation = dilation
        self.num_classes = num_classes

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.in_planes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            (B, 3, H, W) Input data

        Returns
        -------
        output : torch.Tensor
            (B, C, H, W) Predicted logits
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    ###########################
    # PyTorch Lightning Stuff #
    ###########################

    def _log_common(self, result, split, logits, target, loss):
        pred = torch.argmax(logits, dim=-1)
        result.log(f"{split}/loss", loss, prog_bar=True)
        try:
            result.log(f"{split}/acc", accuracy(pred, target), prog_bar=True)
        except RuntimeError:
            # see: https://github.com/PyTorchLightning/pytorch-lightning/issues/3006
            pass

    def _compute_loss(self, pred, target):
        return F.cross_entropy(pred, target)

    def training_step(self, batch, batch_nb):
        """"""

        x, y = batch
        logits = self(x)
        self.last_logits = logits
        loss = self._compute_loss(logits, y)

        # self.logger.experiment.add_image  # TODO: probably in a hook

        result = pl.TrainResult(minimize=loss)
        self._log_common(result, "train", logits, y, loss)

        return result

    def validation_step(self, batch, batch_nb):
        """"""

        # OPTIONAL
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)

        result = pl.EvalResult(checkpoint_on=loss)
        self._log_common(result, "val", logits, y, loss)

        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_callbacks(self):
        """ Moves trainer callback declaration into the model so the same
        training script can be shared across experiments.

        This is not standard pytorch-lightning

        Returns
        -------
        callbacks : list
            List of callback objects to initialize the Trainer object with.
        """
        from pugh_torch.callbacks import TensorBoardAddClassification
        callbacks = [
                TensorBoardAddClassification(),
                ]
        return callbacks

    def train_dataloader(self):
        transform = A.Compose(
            [
                A.SmallestMaxSize(256),
                A.RandomCrop(224, 224),
                A.HorizontalFlip(),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # this is RGB order.
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
        
        dataset = pt.datasets.get("classification", self.cfg.dataset.name)("train", transform=transform)
        loader = DataLoader(dataset, shuffle=True, pin_memory=self.cfg.dataset.pin_memory, num_workers=self.cfg.dataset.num_workers, batch_size=self.cfg.dataset.batch_size)
        return loader

    def val_dataloader(self):
        transform = A.Compose(
            [
                A.SmallestMaxSize(256),
                A.CenterCrop(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # this is RGB order.
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

        dataset = pt.datasets.get("classification", self.cfg.dataset.name)("val", transform=transform)
        loader = DataLoader(dataset, shuffle=False, pin_memory=self.cfg.dataset.pin_memory, num_workers=self.cfg.dataset.num_workers, batch_size=self.cfg.dataset.batch_size)
        return loader
