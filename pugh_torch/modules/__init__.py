from .conv import conv3x3, conv1x1

try:
    import pytorch_lightning
except ImportError:
    pass
else:
    from .lightning_module import LightningModule
