"""Datasets used for the project."""

from datasets import Dataset, arrow_dataset, load_dataset  # type: ignore
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mantis.configs.config import Config
from mantis.configs.data import DatasetConfig, DatasetName
from mantis.data.transforms import get_transforms, hf_transform_caltech256


def setup_dataset(config: DatasetConfig, device: int) -> tuple[Dataset, Dataset]:
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


def get_distributed_dataloader(
    ds: Dataset,
    rank: int,
    config: Config,
    *,
    is_train: bool,
) -> DataLoader:
    """Get distributed data loader.

    Args:
        ds: Pytorch or Hugging Face dataset.
        rank: Global rank.
        config: Training configuration.
        is_train: Whether the data loader is for training.

    Returns:
        Distributed data loader for the given global rank.

    """
    sampler = DistributedSampler(
        dataset=ds,  # type: ignore
        num_replicas=config.num_nodes * config.gpus_per_node,
        rank=rank,
        shuffle=is_train,
    )
    return DataLoader(
        ds, # type: ignore
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=True,
    )
