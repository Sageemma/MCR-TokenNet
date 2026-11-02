from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class TokenReasoning(nn.Module):
    """Adaptive 7×7 tokenization, positional encoding, Transformer, global pooling."""

    def __init__(self, channels: int=128, dim: int=128, grid: int=7, layers: int=2, heads: int=4, mlp_ratio: int=4, dropout: float=0.1):
        super().__init__()
        self.grid = grid
        self.projection = nn.Conv2d(channels, dim, 1)
        self.position = nn.Parameter(torch.zeros(1, grid * grid, dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * mlp_ratio, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(features, self.grid)
        tokens = self.projection(pooled).flatten(2).transpose(1, 2) + self.position
        return self.encoder(tokens)
