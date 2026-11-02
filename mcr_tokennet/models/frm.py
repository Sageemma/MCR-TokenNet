from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
import math
from .encoder import ConvBNAct

class CrossLevelAttention(nn.Module):

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError('FRM channels must be divisible by heads')
        self.heads = heads
        self.dim = dim
        self.d = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        b, n, c = target.shape
        shape = lambda x: x.reshape(b, n, self.heads, self.d).transpose(1, 2)
        q = shape(self.q(target))
        k = shape(self.k(source))
        v = shape(self.v(source))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)
        result = (scores.softmax(dim=-1) @ v).transpose(1, 2).contiguous().reshape(b, n, c)
        return self.out(result)

class FRM(nn.Module):
    """8x8 aligned, pairwise attention, learned global fusion, residual α.

    Q from target pooled features; K,V from another encoder level.
    Cross-level outputs average over the remaining two levels.
    """

    def __init__(self, channels: tuple[int, int, int], dim: int=128, grid: int=8, heads: int=4, dropout: float=0.1):
        super().__init__()
        self.dim = dim
        self.grid = grid
        self.projections = nn.ModuleList((nn.Sequential(nn.Conv2d(ch, dim, 1, bias=False), nn.ReLU(inplace=True)) for ch in channels))
        self.attention = CrossLevelAttention(dim, heads)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.level_logits = nn.Parameter(torch.zeros(3))
        self.fuse = nn.Sequential(ConvBNAct(dim, dim, 3), nn.Dropout2d(p=dropout))

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]):
        if len(features) != 3:
            raise ValueError('FRM expects three hierarchical features')
        pooled = [F.adaptive_avg_pool2d(proj(feat), self.grid) for proj, feat in zip(self.projections, features)]
        seq = [x.flatten(2).transpose(1, 2) for x in pooled]
        recalibrated = []
        for i, x in enumerate(seq):
            updates = [self.attention(x, seq[j]) for j in range(3) if j != i]
            update = torch.stack(updates).mean(dim=0)
            recalibrated.append(pooled[i] + self.alpha * update.transpose(1, 2).reshape_as(pooled[i]))
        weights = self.level_logits.softmax(dim=0)
        mixed = sum((weights[i] * recalibrated[i] for i in range(3)))
        return self.fuse(mixed)
