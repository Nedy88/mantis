"""SimpleTrainer for single interation with fixed grid patches."""


import pydantic
import torch.nn.functional as F  # noqa: N812
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from mantis.configs.config import Config
from mantis.data.datasets import get_distributed_dataloader, setup_dataset
from mantis.model.patch_extractor import extract_grid_patches
from mantis.model.simple_model import SimpleModel
from mantis.optimization.warmup_cosine_scheduler import WarmupCosineScheduler
from mantis.summary import print_table_with_param_counts


class TrainingState(pydantic.BaseModel):
    """Store the training state for the trainer."""

    epoch: int = 0
    global_step: int = 0
    global_iteration: int = 0
    total_global_steps: int | None = None

    @pydantic.field_validator("epoch")
    @classmethod
    def epoch_must_be_non_negative(cls, v: int) -> int:
        """Validate the epoch."""
        if v < 0:
            msg = f"Epoch must be non-negative. Got {v}."
            raise ValueError(msg)
        return v


class SimpleTrainer:
    """SimpleTrainer for single interation with fixed grid patches."""

    def __init__(self, config: Config, local_rank: int, global_rank: int) -> None:
        """Init."""
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.config = config
        self.training_state = TrainingState()
        self.model = SimpleModel(config).cuda(self.local_rank)
        self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning.lr,
            weight_decay=config.learning.weight_decay,
        )
        if self.is_global_zero:
            print_table_with_param_counts(self.model.module, title="Model Parameters")

        # Prepare the data loaders
        train_ds, val_ds = setup_dataset(config.dataset, device=self.local_rank)
        self.train_loader = get_distributed_dataloader(
            train_ds,
            self.global_rank,
            config,
            is_train=True,
        )
        self.val_loader = get_distributed_dataloader(
            val_ds,
            self.global_rank,
            config,
            is_train=False,
        )

        # Learning summary
        eff_batch_size = config.batch_size * self.world_size * config.learning.accum_iter
        steps_per_epoch = len(train_ds) // eff_batch_size
        self.training_state.total_global_steps = steps_per_epoch * config.epochs
        self.lr_scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.learning.warmup_epochs * steps_per_epoch,
            max_steps=self.training_state.total_global_steps,
            min_lr=config.learning.min_lr,
        )

    def train(self) -> None:
        """Train the model."""
        for epoch in range(self.config.epochs):
            self.set_epoch(epoch)
            self.train_epoch()
            break

    def train_epoch(self) -> None:
        """Run an epoch of training."""
        self.model.train()
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.training_state.epoch}"):
            imgs = batch["image"].cuda(self.local_rank)  # (B, 3, H, W)
            labels = batch["label"].cuda(self.local_rank)  # (B,)
            self.update_iterations()
            # Sample patches with the corresponding locations
            patches, locs = extract_grid_patches(imgs, self.config.patch_size)
            logits = self.model(patches, locs) # (B, num_classes)
            loss = F.cross_entropy(logits, labels)
            print(f"Loss: {loss.item()}")
            loss /= self.config.learning.accum_iter
            loss.backward()

            if self.is_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
            break

    def update_iterations(self) -> None:
        """Increase the interation count."""
        self.training_state.global_iteration += 1
        if self.is_step:
            self.training_state.global_step += 1

    @property
    def is_global_zero(self) -> bool:
        """Check if the global rank is zero."""
        return self.global_rank == 0

    @property
    def is_step(self) -> bool:
        """Check if the current iteration is a step."""
        return self.training_state.global_iteration % self.config.learning.accum_iter == 0

    @property
    def world_size(self) -> int:
        """Total number of processes (nodes * gpus per node)."""
        return self.config.num_nodes * self.config.gpus_per_node

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch."""
        self.training_state.epoch = epoch
        assert isinstance(self.train_loader.sampler, DistributedSampler)
        assert isinstance(self.val_loader.sampler, DistributedSampler)
        self.train_loader.sampler.set_epoch(epoch)
        self.val_loader.sampler.set_epoch(epoch)
