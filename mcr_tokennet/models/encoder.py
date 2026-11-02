from __future__ import annotations

import torch
from torch import nn

from .smr import SMR


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResidualBottleneck(nn.Module):
    """Pointwise expansion, spatial filtering, and residual projection."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=1, stride=stride),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        reduction: int,
        kernels: tuple[int, ...],
        bottleneck_widths: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if blocks < 1:
            raise ValueError('At least one block is required')
        if bottleneck_widths is not None and len(bottleneck_widths) != blocks:
            raise ValueError('A bottleneck width is required for each stage block')
        layers = []
        for index in range(blocks):
            src = in_channels if index == 0 else out_channels
            stride = 2 if index == 0 else 1
            if bottleneck_widths is None:
                layers.append(ConvBNAct(src, out_channels, stride=stride))
            else:
                layers.append(ResidualBottleneck(src, out_channels, bottleneck_widths[index], stride))
        self.conv = nn.Sequential(*layers)
        self.smr = SMR(out_channels, reduction, kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.smr(self.conv(x))
