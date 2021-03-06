import torch
import torchvision

from . import ROOT_MODELS_PATH, load_state_dict_from_url
from ..modules import LoadStateDictMixin

from torchvision.models.resnet import model_urls, Bottleneck


class ResNet(LoadStateDictMixin, torchvision.models.ResNet):
    """Just mixes in our ``load_state_dict`` method that allows
    easier fine-tuning/transfer learning.
    """


""" Copied from torchvision.models.resnet
"""


def _resnet(arch, block, layers, pretrained, progress, strict=True, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        if kwargs.get("num_classes", 1000) != 1000:
            # By changing the number of output classes, its strongly implied
            # that we don't want to use strict loading.
            strict = False
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=strict)
    return model


def resnet18(pretrained=False, progress=True, strict=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )


def resnet34(pretrained=False, progress=True, strict=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )


def resnet50(pretrained=False, progress=True, strict=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )


def resnet101(pretrained=False, progress=True, strict=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )


def resnet152(pretrained=False, progress=True, strict=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152",
        Bottleneck,
        [3, 8, 36, 3],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, strict=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, strict=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, strict=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, strict=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        strict=strict,
        **kwargs
    )
