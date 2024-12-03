"""Configuration."""


import pydantic

from mantis.configs.data import DatasetConfig
from mantis.configs.transformer import TransformerConfig


class Config(pydantic.BaseModel):
    """The main training configuration."""

    transformer: TransformerConfig
    dataset: DatasetConfig
    patch_size: int

    class Config:  # noqa: D106
        use_enum_values = True
