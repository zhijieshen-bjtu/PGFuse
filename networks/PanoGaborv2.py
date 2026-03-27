import math
from typing import Any

import torch
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module


class GaborConv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        super().__init__()

        self.is_calculated = False

        self.conv_layer = Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size

        # small addition to avoid division by zero
        self.delta = 1e-3
        self.height = out_channels
        self.in_channels = in_channels

        # freq, theta, sigma are set up according to S. Meshgini,
        # A. Aghagolzadeh and H. Seyedarabi, "Face recognition using
        # Gabor filter bank, kernel principal component analysis
        # and support vector machine"
        # Initialize Gabor parameters
        self.freq = Parameter(torch.Tensor(out_channels, in_channels))
        #self.theta = Parameter(torch.Tensor(out_channels, in_channels))
        self.sigma = Parameter(torch.Tensor(out_channels, in_channels))
        #self.psi = Parameter(torch.Tensor(out_channels, in_channels))

        self.init_parameters()
        self.theta = Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True,
        )
        self.psi = Parameter(
            math.pi * torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.x0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
        )
        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ]
        )
        self.y = Parameter(self.y)
        self.x = Parameter(self.x)

        self.weight = Parameter(
            torch.empty(self.conv_layer.weight.shape, requires_grad=True),
            requires_grad=True,
        )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def init_parameters(self):
        mid_point = self.height // 2
        scale_factor = torch.linspace(0, 1, mid_point)
        random_exponents = -torch.randint(0, 5, (mid_point, self.in_channels))

        # Initialize freq with a function of position; higher at edges
        self.freq.data[:mid_point, :] = (math.pi / 2) * (math.sqrt(2) ** random_exponents) * (
                    1 + scale_factor.unsqueeze(1))

        # Mirror parameters for the other half
        self.freq.data[mid_point:, :] = torch.flip(self.freq.data[:mid_point, :], [0])
        self.sigma.data = math.pi / (self.freq.data + 0.1)  # Avoid division by zero

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        theta = self.theta.view(-1, 1, 1)
        sigma = self.sigma.view(-1, 1, 1)
        freq = self.freq.view(-1, 1, 1)
        psi = self.psi.view(-1, 1, 1)

        rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
        roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

        g = torch.exp(-0.5 * (rotx ** 2 + roty ** 2) / (sigma ** 2))
        g *= torch.cos(freq * rotx + psi)
        g /= (2 * math.pi * sigma ** 2)

        self.conv_layer.weight.data = g.view_as(self.conv_layer.weight.data)

    def _forward_unimplemented(self, *inputs: Any):
        """
        code checkers makes implement this method,
        looks like error in PyTorch
        """
        raise NotImplementedError