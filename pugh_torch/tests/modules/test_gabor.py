from pugh_torch.modules import GaborConv2d
import matplotlib.pyplot as plt
import torch

def fake_rgb_data():
    data = torch.rand((1, 3, 224, 224))
    return data

def test_gabor_conv_2d():
    conv_kwargs = {
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": 9,
            "stride": 1,
            }
    gabor_kwargs = {
            "frequency": 0.05,  # In radians?
            }

    conv = GaborConv2d(
            **conv_kwargs,
            **gabor_kwargs,
            )
    pass
