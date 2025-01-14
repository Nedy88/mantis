"""Extracting patches from images given locations."""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


@torch.no_grad()
def extract_equal_zoom_patches(
    images: Tensor,
    locs: Tensor,
    z: float,
    patch_size: int,
) -> Tensor:
    """Extract equal zoom patches from a batch of images.

    Extract single patch per image at the same zoom level across the batch.

    Args:
        images: Batch of images of shape (B, 3, H, W).
        locs: Batch of locations of shape (B, 2) containing (x, y).
        z: The zoom level for all patches.
        patch_size: The size of each output patch is (3, patch_size, patch_size).

    Returns:
        Patches of shape (B, 3, patch_size, patch_size).

    """
    B, _, H, W = images.shape  # noqa: N806
    aspect_ratio = W / H
    y_coords = torch.linspace(-1, 1, patch_size)
    x_coords = torch.linspace(-1, 1, patch_size)
    if aspect_ratio > 1:
        # Image in landscape orientation.
        x_coords = x_coords * aspect_ratio
    else:
        # Image in portrait orientation.
        y_coords = y_coords / aspect_ratio
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1)  # (P, P, 2)
    # Apply zoom level
    sampling_grid /= 2**z
    sampling_grid = sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)

    # Change the centers according to the locs
    sampling_grid = sampling_grid + locs[:, None, None, :]

    return F.grid_sample(
        images,
        sampling_grid,
        mode="bilinear",
        align_corners=False,
    )


@torch.no_grad()
def extract_grid_patches(
    images: Tensor,
    patch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Extract ViT-like grid of equal sized patches from a batch of images.

    Args:
        images: Batch of images of shape (B, 3, H, W).
        patch_size: The size of each output patch is (3, patch_size, patch_size).
        device: Device to put on the created tensors.

    Returns:
        Tuple of Tensors. The first one is the extracted patches of shape (B, M, 3, P, P),
        and the second is the locations of the patches of shape (B, M, 2).

    """
    B, C, H, W = images.shape  # noqa: N806
    assert H % patch_size == 0, "Image height must be divisible by patch size"
    assert W % patch_size == 0, "Image width must be divisible by patch size"
    Hp = H // patch_size  # noqa: N806
    Wp = W // patch_size  # noqa: N806
    patches = images.view(B, C, Hp, patch_size, Wp, patch_size)  # (B, C, Hp, P, Wp, P)
    patches = patches.permute(0, 2, 4, 1, 3, 5)  # (B, Hp, Wp, C, P, P)
    patches = patches.reshape(B, -1, C, patch_size, patch_size)  # (B, M, C, P, P)
    z = math.log2(min(Hp, Wp))  # Fixed zoom level
    patch_width = 2 * patch_size / W  # Width in the space of relative coordinates between -1 and 1
    patch_height = (
        2 * patch_size / H
    )  # Height in the space of relative coordinates between -1 and 1
    xs = torch.linspace(-1 + patch_width / 2, 1 - patch_width / 2, Wp).to(device)
    ys = torch.linspace(-1 + patch_height / 2, 1 - patch_height / 2, Hp).to(device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (Hp, Wp)
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1)  # (Hp, Wp, 2)
    sampling_grid = sampling_grid.view(-1, 2).unsqueeze(0).expand(B, -1, -1)  # (B, M, 2)
    locs = torch.ones(B, Hp * Wp, 3).to(device) * z  # (B, M, 3)
    locs[:, :, :2] = sampling_grid
    return patches, locs
