from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .encoder import ConvBNAct, EncoderStage
from .frm import FRM
from .token_reasoning import TokenReasoning

@dataclass(frozen=True)
class ArchitectureConfig:
    stem_channels: int = 32
    stage_channels: tuple[int, int, int] = (64, 128, 256)
    blocks_per_stage: int = 2
    stage2_bottleneck_widths: tuple[int, int] = (152, 208)
    stage3_bottleneck_widths: tuple[int, int] = (400, 448)
    smr_reduction: int = 8
    smr_kernels: tuple[int, int, int] = (3, 5, 7)
    frm_channels: int = 128
    frm_grid: int = 8
    frm_heads: int = 4
    token_grid: int = 7
    token_dim: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1

class MCRTokenNet(nn.Module):
    """SMR encoder → pooled-grid FRM → compact token classifier."""

    def __init__(self, num_classes: int, cfg: ArchitectureConfig | None=None):
        super().__init__()
        if num_classes < 2:
            raise ValueError('At least two classes are required')
        self.cfg = cfg or ArchitectureConfig()
        c = self.cfg
        self.stem = nn.Sequential(ConvBNAct(3, c.stem_channels), ConvBNAct(c.stem_channels, c.stem_channels), nn.MaxPool2d(2))
        stages = []
        input_channels = c.stem_channels
        for index, channels in enumerate(c.stage_channels):
            widths = (
                c.stage2_bottleneck_widths if index == 1
                else c.stage3_bottleneck_widths if index == 2
                else None
            )
            stages.append(EncoderStage(
                input_channels, channels, c.blocks_per_stage,
                c.smr_reduction, c.smr_kernels, widths,
            ))
            input_channels = channels
        self.stages = nn.ModuleList(stages)
        self.frm = FRM(c.stage_channels, c.frm_channels, c.frm_grid, c.frm_heads, c.dropout)
        self.token_reasoning = TokenReasoning(c.frm_channels, c.token_dim, c.token_grid, c.transformer_layers, c.transformer_heads, c.mlp_ratio, c.dropout)
        self.head = nn.Sequential(nn.LayerNorm(c.token_dim), nn.Linear(c.token_dim, num_classes))

    def forward_features(self, x: torch.Tensor) -> dict:
        x = self.stem(x)
        hierarchy = []
        for stage in self.stages:
            x = stage(x)
            hierarchy.append(x)
        fused = self.frm(hierarchy)
        tokens = self.token_reasoning(fused)
        return {'hierarchy': hierarchy, 'frm': fused, 'tokens': tokens, 'representation': tokens.mean(dim=1)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)['representation'])

def parameter_count(model: nn.Module) -> int:
    return sum((parameter.numel() for parameter in model.parameters()))
