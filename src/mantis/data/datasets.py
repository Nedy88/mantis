"""Datasets used for the project."""

from datasets import Dataset, arrow_dataset, load_dataset  # type: ignore

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
            )
            assert isinstance(train_ds, arrow_dataset.Dataset)
            train_ds.set_transform(transforms)
            val_ds = load_dataset(
                config.hf_dataset,
                split="validation",
                cache_dir=str(config.root),
            )
            assert isinstance(val_ds, arrow_dataset.Dataset)
            val_ds.set_transform(transforms)
            return (train_ds, val_ds)
        case _:
            msg = f"Unknown dataset: {config.name}"
            raise ValueError(msg)
