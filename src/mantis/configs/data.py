"""Configuration for the datasets."""

from enum import Enum
from pathlib import Path

import pydantic


class DatasetName(str, Enum):
    """Datasets."""

    caltech256 = "caltech256"


class DatasetConfig(pydantic.BaseModel):
    """Configuration for the dataset."""

    name: DatasetName
    hf_dataset: str | None
    root: Path
    img_width: int
    img_height: int
    num_classes: int | None
    num_workers: int

    class Config:  # noqa: D106
        use_enum_values = True
