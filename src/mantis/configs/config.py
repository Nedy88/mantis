"""Configuration."""


import pydantic

from mantis.configs.data import DatasetConfig
from mantis.configs.transformer import TransformerConfig


class Config(pydantic.BaseModel):
    """The main training configuration."""

    transformer: TransformerConfig
    dataset: DatasetConfig

    class Config:  # noqa: D106
        use_enum_values = True
