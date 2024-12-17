"""SimpleTrainer for taining on TPUs."""

import torch_xla.core.xla_model as xm  # type: ignore[reportMissingTypeStubs]
import torch_xla.distributed.parallel_loader as pl  # type: ignore[reportMissingTypeStubs]
from torch_xla import runtime as xr  # type: ignore[reportMissingTypeStubs]

from mantis.configs.config import Config
from mantis.data.datasets import (
    get_per_device_batch_size,
    prepare_distributed_dataloaders,
)


class SimpleTrainerXla:
    """SimpleTrainer for single iteration with fixed grid patches."""

    def __init__(self, config: Config) -> None:
        """Init."""
        self.device = xm.xla_device()
        batch_size = get_per_device_batch_size(config)
        train_loader, val_loader = prepare_distributed_dataloaders(
            config,
            xr.global_ordinal(),
            batch_size,
        )
        self.train_loader = pl.MpDeviceLoader(train_loader, self.device)
        self.val_loader = pl.MpDeviceLoader(val_loader, self.device)

    def train(self) -> None:
        """Train the model."""
        print(f"Training from rank: {xr.global_ordinal()} out of {xr.world_size()}")
