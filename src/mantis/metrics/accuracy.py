"""Compute accuracy of a classification method."""

import torch
from torch import Tensor

import mantis.distributed as D  # noqa: N812


class Accuracy:
    """Compute multi-class accuracy for a classification task."""

    def __init__(self, topk: int) -> None:
        """Init."""
        self.topk = topk
        self.correct: list[Tensor] = []
        self.total: list[Tensor] = []
        self.eps = 1e-6

    def __call__(self, logits: Tensor, labels: Tensor) -> None:
        """Store stats for the current batch.

        Args:
            logits: Output of the model (B, num_classes).
            labels: Ground truth labels (B,).

        """
        _, topk_indices = torch.topk(logits, self.topk, dim=1)  # (B, topk)
        correct = torch.any(topk_indices == labels.unsqueeze(1), dim=1)  # (B,)
        self.correct.append(correct.sum().float().squeeze())
        self.total.append(torch.tensor(labels.shape[0]).to(logits).squeeze())

    def reset(self) -> None:
        """Reset the stats."""
        self.correct = []
        self.total = []

    def compute(self) -> Tensor:
        """Compute the topk accuracy."""
        correct = D.all_gather_scalar(self.correct).sum()
        total = D.all_gather_scalar(self.total).sum()
        if total < self.eps:
            return torch.tensor(0.0).to(correct)
        return correct / total

