"""Test the DropPath module."""

import torch

from mantis.model.transformer import DropPath


def test_drop_path(drop_path_prob: float) -> None:
    """Test the DropPath module."""
    batch_size = 16
    n = 4
    d = 512
    drop_path_prob = 0.5
    eps = 1e-6
    inp = torch.randn(batch_size, n, d)
    drop_path = DropPath(drop_path_prob)
    output = drop_path(inp)
    assert inp.shape == output.shape
    all_zeros = torch.all(output.flatten(1) == 0, dim=1)
    mask = ~all_zeros
    assert (
        torch.abs(inp[mask].flatten() - output[mask].flatten() * (1.0 - drop_path_prob)).max() < eps
    )

def test_drop_path_all() -> None:
    """Test dropping all elements."""
    batch_size = 16
    n = 4
    d = 512
    eps = 1e-6
    inp = torch.randn(batch_size, n, d)
    drop_path = DropPath(1.)
    out = drop_path(inp)
    assert torch.all(out.abs() < eps)

def test_drop_path_none() -> None:
    """Test dropping no elements."""
    batch_size = 16
    n = 4
    d = 512
    eps = 1e-6
    inp = torch.randn(batch_size, n, d)
    drop_path = DropPath(0.)
    out = drop_path(inp)
    assert torch.abs(inp - out).max() < eps
