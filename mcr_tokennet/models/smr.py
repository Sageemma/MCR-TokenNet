from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class SMR(nn.Module):
    """Channel interaction + multiscale depthwise spatial response, residual gate γ.

    """

    def __init__(self, channels: int, reduction: int=8, kernels: tuple[int, ...]=(3, 5, 7)):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.branches = nn.ModuleList((nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False) for k in kernels))
        self.mix = nn.Sequential(nn.Conv2d(channels * len(kernels), channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x * self.channel(x)
        update = self.mix(torch.cat([branch(scaled) for branch in self.branches], dim=1))
        return x + self.gamma * update
