"""SimpleTrainer for taining on TPUs."""

from functools import partial

import torch
import torch.nn.functional as F  # noqa: N812
import torch_xla
import torch_xla.core.xla_model as xm  # type: ignore[reportMissingTypeStubs]
import torch_xla.distributed.parallel_loader as pl  # type: ignore[reportMissingTypeStubs]
from torch import Tensor, nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch_xla import runtime as xr  # type: ignore[reportMissingTypeStubs]
from tqdm import tqdm

from mantis.configs.config import Config
from mantis.data.datasets import (
    get_per_device_batch_size,
    prepare_distributed_dataloaders,
)
from mantis.model.patch_extractor import extract_grid_patches
from mantis.model.simple_model import DummyModel, SimpleModel
from mantis.optimization.warmup_cosine_scheduler import WarmupCosineScheduler
from mantis.summary import (
    print_table_with_param_counts,
    print_training_step_stats_xla,
)
from simple_trainer import TrainingState


class SimpleTrainerXla:
    """SimpleTrainer for single iteration with fixed grid patches."""

    def __init__(self, config: Config) -> None:
        """Init."""
        self.device = xm.xla_device()
        self.config = config
        self.training_state = TrainingState()
        self.validate_config()

        # Setup dataloaders
        batch_size = get_per_device_batch_size(config)
        self.train_loader, self.val_loader = prepare_distributed_dataloaders(
            config,
            xr.global_ordinal(),
            batch_size,
        )
        # self.train_loader = pl.MpDeviceLoader(train_loader, self.device)
        # self.val_loader = pl.MpDeviceLoader(val_loader, self.device)
        xm.master_print("Done setting the dataloaders.")  # type: ignore

        # Setup the model and optimizer for DDP
        xm.master_print("Setting the model.")  # type: ignore
        # model = SimpleModel(config).to(self.device)
        self.model = DummyModel(config).to(self.device)
        # model = nn.Sequential(nn.Conv2d(3, 4, 4), nn.ReLU()).to(self.device)
        # model = nn.Sequential(nn.Conv2d(3, 4, 10), nn.ReLU(), nn.AvgPool2d(10)).to(self.device)
        # TODO: Revert this after test. It's extremely slow
        # xm.broadcast_master_param(model)
        xm.master_print("After broadcasting params.")  # type: ignore
        # self.model = DistributedDataParallel(model, gradient_as_bucket_view=True)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning.lr)
        # self.optimizer = optim.AdamW(
        #     self.model.module.get_param_groups(config.learning.weight_decay),
        #     lr=config.learning.lr,
        #     weight_decay=config.learning.weight_decay,
        # )
        if xr.global_ordinal() == 0:
            print_table_with_param_counts(self.model.module, title="Model Parameters")

        # Learning summary
        steps_per_epoch = len(self.train_loader)
        self.training_state.total_global_steps = steps_per_epoch * config.epochs
        if xr.global_ordinal() == 0:
            print_training_step_stats_xla(config, steps_per_epoch)

        # Warmup Cosine Scheduler for the LR
        self.lr_scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.learning.warmup_epochs * steps_per_epoch,
            max_steps=self.training_state.total_global_steps,
            min_lr=config.learning.min_lr,
        )
        self.patch_extractor = partial(extract_grid_patches, device=self.device)
        # self.compiled_step = torch_xla.compile(
        #     self.step_fn,
        #     full_graph=True,
        #     name="simple_model_step_fn",
        # )

    def train(self) -> None:
        """Train the model."""
        for epoch in range(self.config.epochs):
            self.set_epoch(epoch)
            self.train_epoch()


    def train_epoch(self) -> None:
        """Run an epoch of training."""
        self.model.train()
        xm.master_print(f"Starting epoch {self.training_state.epoch}")
        # for step, batch in enumerate(self.train_loader):
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.training_state.epoch}"):
            # self.update_iterations()
            imgs = batch["image"]
            labels = batch["label"]
            with torch_xla.step():
                imgs = imgs.to(self.device)  # (B, 3, H, W)
                labels = labels.to(self.device)  # (B,)
                _ = self.step_fn(imgs, labels)

            # patches, locs = extract_grid_patches(imgs, self.config.patch_size)
            # logits = self.model(patches, locs)
            # loss = F.cross_entropy(logits, labels)
            # loss /= self.config.learning.accum_iter
            # loss.backward()
            # if self.is_step:
            #     xm.optimizer_step(self.optimizer)
            #     self.optimizer.zero_grad()
            #     self.lr_scheduler.step()
            #     # TODO(Nedyalko): Log LRs to wandb

    def step_fn(self, images: Tensor, labels: Tensor) -> Tensor:
        """Perform a training step."""
        self.optimizer.zero_grad()
        patches, locs = self.patch_extractor(images, self.config.patch_size)
        # output = self.model(patches[:, 0, ...]) # (B, 4, h, w)
        logits = self.model(patches, locs)
        loss = F.cross_entropy(logits, labels)
        # loss = (logits.flatten(1).mean(dim=1) - labels).mean()
        loss.backward()
        # self.optimizer.step()
        xm.optimizer_step(self.optimizer)
        return loss

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch."""
        self.training_state.epoch = epoch
        assert isinstance(self.train_loader.sampler, DistributedSampler)
        assert isinstance(self.val_loader.sampler, DistributedSampler)
        self.train_loader.sampler.set_epoch(epoch)
        self.val_loader.sampler.set_epoch(epoch)

    def update_iterations(self) -> None:
        """Increase iterations count."""
        self.training_state.global_iteration += 1
        if self.is_step:
            self.training_state.global_step += 1

    @property
    def is_step(self) -> bool:
        """Is the current iteration a step."""
        return self.training_state.global_iteration % self.config.learning.accum_iter == 0

    def validate_config(self) -> None:
        """Validate the configuration."""
        assert self.config.learning.accum_iter == 1, (
            "Current we only support accum_iter being 1, but it is"
            f" {self.config.learning.accum_iter}",
        )
