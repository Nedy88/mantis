"""Cosine LR scheduler with linear warmup."""

import math

from torch import optim


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """Cosit LR scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr: float,
    ) -> None:
        """Init."""
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        super().__init__(optimizer)


    def get_lr(self) -> list[float]:
        """Get the learning rate."""
        if self._step_count < self.warmup_steps:
            # Linear warmup
            return [
                self.min_lr + (base_lr - self.min_lr) * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        return [
            self.min_lr
            + 0.5 * (base_lr - self.min_lr)
            * (1 + math.cos(math.pi * self._step_count / self.max_steps))
            for base_lr in self.base_lrs
        ]
