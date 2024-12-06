"""Helpers for distributed training."""

import torch
import torch.distributed as dist
from torch import Tensor


def all_gather(tensor: Tensor) -> list[Tensor]:
    """All gather the tensor to all processes."""
    gather_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, tensor)
    return gather_list

def all_gather_scalar(values: list[Tensor]) -> Tensor:
    """All gather a scalar metric.

    Args:
        values: List of tensors with ndim == 1, containing the metric values in a batch.

    Returns:
        Tensor containing all values from all processes, with ndim == 1.

    """
    stacked_values = torch.stack(values)  # (N,)
    gathered_values = all_gather(stacked_values)  # [(N,)]
    return torch.cat(gathered_values, dim=0)  # (N*world_size,)
