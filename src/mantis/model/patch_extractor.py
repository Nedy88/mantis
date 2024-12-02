"""Extracting patches from images given locations."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


@torch.no_grad()
def extract_patches(images: Tensor, locs: Tensor, patch_size: int) -> Tensor:
    """Extract patches from images and given locations.

    Args:
        images: Batch of image of shape (B, 3, H, W).
        locs: Batch of locations of shape (B, M, 3) containing (x, y, z).
        patch_size: The size of each output patch is (3, patch_size, patch_size).

    Returns:
        Patches of shape (B, M, 3, patch_size, patch_size).

    """
    B, _, H, W = images.shape  # noqa: N806
    M = locs.shape[1]
    min_dim = min(H, W)

    # Calculate the patch sizes based on the z value
    patch_sizes = min_dim / (2 ** locs[..., 2])
    patch_sizes = torch.clamp(patch_sizes, min=1)

    # Create messh grid for sampling
    y_coords = torch.linspace(-1, 1, patch_size)
    x_coords = torch.linspace(-1, 1, patch_size)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    sampling_grid = torch.stack([grid_x, grid_y], dim=-1)  # (P, P, 2)

    scale = (patch_sizes / min_dim).view(B, M, 1, 1, 1)
    sampling_grid = sampling_grid.view(1, 1, patch_size, patch_size, 2).expand(B, M, -1, -1, -1)
    sampling_grid = sampling_grid * scale # (B, M, P, P, 2)

    center_x = (2 * locs[..., 0] / W - 1).view(B, M, 1, 1)
    center_y = (2 * locs[..., 1] / H - 1).view(B, M, 1, 1)

    sampling_grid[..., 0] = sampling_grid[..., 0] + center_x
    sampling_grid[..., 1] = sampling_grid[..., 1] + center_y

    images = images.view(B, 1, 3, H, W).expand(-1, M, -1, -1, -1)
    images = images.reshape(B * M, 3, H, W)
    sampling_grid = sampling_grid.view(B * M, patch_size, patch_size, 2)

    patches = F.grid_sample(
        images,
        sampling_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return patches.view(B, M, 3, patch_size, patch_size)
