"""Datasets used for the project."""

from datasets import Dataset, arrow_dataset, load_dataset  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mantis.configs.config import Config
from mantis.configs.data import DatasetConfig, DatasetName
from mantis.data.transforms import get_transforms, hf_transform_caltech256


def setup_dataset(config: DatasetConfig) -> tuple[Dataset, Dataset]:
    """Load and setup datasets."""
    match config.name:
        case DatasetName.caltech256:
            assert config.hf_dataset is not None
            transforms = get_transforms(config)
            transforms = hf_transform_caltech256(transforms)
            train_ds = load_dataset(
                config.hf_dataset,
                split="train",
                cache_dir=str(config.root),
            ).with_format("torch")
            assert isinstance(train_ds, arrow_dataset.Dataset)
            train_ds.set_transform(transforms)
            val_ds = load_dataset(
                config.hf_dataset,
                split="validation",
                cache_dir=str(config.root),
            ).with_format("torch")
            assert isinstance(val_ds, arrow_dataset.Dataset)
            val_ds.set_transform(transforms)
            return (train_ds, val_ds)
        case _:
            msg = f"Unknown dataset: {config.name}"
            raise ValueError(msg)


def get_per_device_batch_size(config: Config) -> int:
    """Get per device batch size.

    Args:
        config: Global configuration containing effective batch_size.

    Returns:
        Per device batch size for the dataloaders.

    """
    assert config.effective_batch_size % config.world_size == 0, (
        f"Effective batch size {config.effective_batch_size} must be divisible "
        f"by world size {config.world_size}"
    )
    per_process_batch_size = config.effective_batch_size // config.world_size
    assert per_process_batch_size % config.learning.accum_iter == 0, (
        f"Per process batch size {per_process_batch_size} must be divisible "
        f"by accumulation steps {config.learning.accum_iter}"
    )
    return per_process_batch_size // config.learning.accum_iter


def prepare_distributed_dataloaders(
    config: Config,
    rank: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Prepare distributed dataloader from the config."""
    train_ds, val_ds = setup_dataset(config.dataset)
    train_loader = get_distributed_dataloader(
        train_ds,
        rank,
        config=config,
        batch_size=batch_size,
        is_train=True,
    )
    val_loader = get_distributed_dataloader(
        val_ds,
        rank,
        config=config,
        batch_size=batch_size,
        is_train=False,
    )
    return train_loader, val_loader


def get_distributed_dataloader(
    ds: Dataset,
    rank: int,
    *,
    config: Config,
    batch_size: int,
    is_train: bool,
) -> DataLoader:
    """Get distributed data loader.

    Args:
        ds: Pytorch or Hugging Face dataset.
        rank: Global rank.
        config: Configuration.
        batch_size: Per process batch size.
        is_train: Whether the data loader is for training.

    Returns:
        Distributed data loader for the given global rank.

    """
    sampler = DistributedSampler(
        dataset=ds,  # type: ignore
        num_replicas=config.world_size,
        rank=rank,
        shuffle=is_train,
    )
    return DataLoader(
        ds,  # type: ignore
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.dataset.num_workers,
        persistent_workers=config.dataset.num_workers > 0,
        pin_memory=True,
    )
