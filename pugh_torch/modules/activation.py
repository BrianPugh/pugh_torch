""" Easy interface for swapping out activation functions, especially those
that may have different weight initialization methods.

    * weights  <- initialization depends on activation function <----
    * normalization                                                 |
    * activation <---------------------------------------------------

To create a new activation, do the following:
    * Inherit from ActivationModule to register
    * [optional] implement ``init_layer`` method
    * [optional] implement ``init_first_layer`` method
One this is done, your activation function will me available as:
    Activation("myactivationlowercase", **kwargs)

"""

import torch
from torch import nn
import pugh_torch.modules.init as wi

_activation_lookup = {}


class ActivationModule(nn.Module):
    """Only used to automatically register activation functions."""

    def __init_subclass__(cls, **kwargs):
        """Automatic registration stuff"""
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower()
        if name in _activation_lookup:
            raise ValueError(
                f'Activation function "{name}" already exists: {_activation_lookup[name]}'
            )
        _activation_lookup[name] = cls

    @torch.no_grad()
    def init_layer(self, m):
        """
        Override this in child activation function
        """
        pass

    @torch.no_grad()
    def init_first_layer(self, m):
        """
        Override this in child activation function
        """

        return self.init_layer(m)


def Activation(name, init_layers=None, *, first=False, **kwargs):
    """Activation Factory Function

    Parameters
    ----------
    name : str
        Activation function type
    init_layers : nn.Module or list of nn.Module
        Weights that need initialization based on
    kwargs : dict
        Passed along to activation function constructor.
    """

    name = name.lower()

    if init_layers is not None:
        if isinstance(init_layers, nn.Module):
            init_layers = [
                init_layers,
            ]
        assert isinstance(init_layers, list)

    activation_obj = _activation_lookup[name](**kwargs)

    if init_layers:
        for init_layer in init_layers:
            if first:
                init_layer.apply(activation_obj.init_first_layer)
            else:
                init_layer.apply(activation_obj.init_layer)

    return activation_obj


class Sine(ActivationModule):
    """
    Implicit Neural Representations with Periodic Activation Functions
    https://arxiv.org/pdf/2006.09661.pdf
    """

    def __init__(self, frequency=30):
        super().__init__()
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        self.frequency = frequency

    def forward(self, input):
        return torch.sin(self.frequency * input)

    @torch.no_grad()
    def init_layer(self, m):
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(
                -np.sqrt(6 / num_input) / self.frequency,
                np.sqrt(6 / num_input) / self.frequency,
            )

    @torch.no_grad()
    def init_layer(self, m):
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


#################################
# torch.nn activation functions #
#################################


class ELU(nn.ELU, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            nn.init.normal_(
                m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input)
            )


class Hardshrink(nn.Hardshrink, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m)


class Hardsigmoid(nn.Hardsigmoid, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m)


class Hardtanh(nn.Hardtanh, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m)


class Hardswish(nn.Hardswish, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.he(m, nonlinearity="relu")


class LeakyReLU(nn.LeakyReLU, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.he(m, nonlinearity="leaky_relu")


class LogSigmoid(nn.LogSigmoid, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m)


class MultiheadAttention(nn.MultiheadAttention, ActivationModule):
    pass


class PReLU(nn.PReLU, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.he(m, nonlinearity="leaky_relu")


class ReLU(nn.ReLU, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.he(m, nonlinearity="relu")


class ReLU6(nn.ReLU6, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.he(m, nonlinearity="relu")


class RReLU(nn.RReLU, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.he(m, nonlinearity="leaky_relu")


class SELU(nn.SELU, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


class CELU(nn.CELU, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


class GELU(nn.GELU, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.he(m, nonlinearity="relu")


class Sigmoid(nn.Sigmoid, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m)


class Softplus(nn.Softplus, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.he(m, nonlinearity="relu")


class Softshrink(nn.Softshrink, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m)


class Softsign(nn.Softsign, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m)


class Tanh(nn.Tanh, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m, gain=nn.init.calculate_gain("tanh"))


class Tanhshrink(nn.Tanhshrink, ActivationModule):
    @torch.no_grad()
    def init_layer(self, m):
        wi.xavier(m)


class Threshold(nn.Threshold, ActivationModule):
    pass
