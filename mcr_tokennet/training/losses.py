"""Class-balanced focal classification loss."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class ClassBalancedFocalLoss(nn.Module):
    """Effective-number class weights with smoothed focal targets."""

    def __init__(self, class_counts: list[int], beta: float=0.999, gamma: float=2.0, epsilon: float=0.05):
        super().__init__()
        if min(class_counts) <= 0:
            raise ValueError('Every training class requires >0 examples')
        n = torch.as_tensor(class_counts, dtype=torch.float64)
        weight = (1 - beta) / (1 - torch.pow(torch.tensor(beta, dtype=torch.float64), n))
        weight = weight / weight.mean()
        self.register_buffer('weights', weight.float())
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        if logits.shape[-1] != self.weights.numel():
            raise ValueError('Class count does not match logits')
        k = logits.shape[-1]
        probabilities = logits.softmax(dim=1)
        q = F.one_hot(labels, k).to(logits.dtype) * (1 - self.epsilon) + self.epsilon / k
        weighted = self.weights.to(dtype=logits.dtype, device=logits.device).unsqueeze(0) * q
        focal = torch.pow((1 - probabilities).clamp(min=0), self.gamma)
        return -(weighted * focal * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
