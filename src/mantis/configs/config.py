"""Configuration."""

import pydantic

from mantis.configs.data import DatasetConfig
from mantis.configs.transformer import TransformerConfig


class Learning(pydantic.BaseModel):
    """Learning hyperparams."""

    lr: float
    accum_iter: int
    min_lr: float
    warmup_epochs: int
    weight_decay: float


class Config(pydantic.BaseModel):
    """The main training configuration."""

    transformer: TransformerConfig
    dataset: DatasetConfig
    learning: Learning
    patch_size: int
    float32_matmul_precision: str
    world_size: int
    batch_size: int
    effective_batch_size: int
    epochs: int
    log_every_n_steps: int

    class Config:  # noqa: D106
        use_enum_values = True
