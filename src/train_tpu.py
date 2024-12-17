"""Training script on TPUs with torch xla."""

import hydra
import torch_xla  # type: ignore[reportMissingTypeStubs]
import torch_xla.core.xla_model as xm  # type: ignore[reportMissingTypeStubs]
from omegaconf import OmegaConf

from mantis.configs.config import Config
from mantis.settings import Secrets


def _mp_fn(index: int, config: Config) -> None:
    print(f"From within index {index} with device: {xm.xla_device()}")
    xm.master_print("Hello") # type: ignore[reportArgumentType]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(config: Config) -> None:
    """Training script."""
    OmegaConf.resolve(config)  # type: ignore
    config = Config(**config)  # type: ignore
    secrets = Secrets()  # type: ignore[reportUnusedVariable]

    torch_xla.launch(_mp_fn, args=(config,))


if __name__ == "__main__":
    train()
