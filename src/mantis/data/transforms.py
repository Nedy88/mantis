"""Data augmentations."""

from collections.abc import Callable
from typing import Any

import torchvision.transforms as tvt  # type: ignore
import torchvision.transforms._functional_pil as f_pil  # type: ignore
from PIL import Image
from torch import nn

from mantis.configs.data import DatasetConfig


class ResizeAndPad(nn.Module):
    """Augmentation to resize and pad the image.

    The goal is to have no aspect ratio distortions.
    """

    def __init__(self, target_size: tuple[int, int]) -> None:
        """Init."""
        super().__init__()
        self.target_size = target_size
        # Aspect ratio is width / height
        self.target_aspect_ratio = target_size[1] / target_size[0]

    def forward(self, img: Image.Image) -> Image.Image:
        """Resize and pad the image."""
        w, h = img.size
        aspect_ratio = w / h
        if aspect_ratio > self.target_aspect_ratio:
            # The image is wider than the target
            new_w = self.target_size[1]
            new_h = int(round(new_w / aspect_ratio))
            diff = self.target_size[0] - new_h
            top, bottom = diff // 2, diff - diff // 2
            padding = (0, top, 0, bottom)
        else:
            # The image is taller than the target
            new_h = self.target_size[0]
            new_w = int(round(new_h * aspect_ratio))
            diff = self.target_size[1] - new_w
            left, right = diff // 2, diff - diff // 2
            padding = (left, 0, right, 0)
        img = img.resize((new_w, new_h))
        return f_pil.pad(img, padding)


def get_transforms(config: DatasetConfig) -> tvt.Compose:
    """Get the augmentations for training and validation."""
    return tvt.Compose(
        [
            ResizeAndPad((config.img_height, config.img_width)),
            tvt.ToTensor(),
        ],
    )


def hf_transform_caltech256(transform: tvt.Compose) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Apply a torchvision transform to hugging face dataset."""

    def impl(examples: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": [transform(image.convert("RGB")) for image in examples["image"]],
            "label": [label - 1 for label in examples["label"]],
        }

    return impl
