"""Training script."""

import hydra
from omegaconf import OmegaConf

from mantis.configs.config import Config
from mantis.data.datasets import setup_dataset


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(config: Config) -> None:
    """Training script."""
    OmegaConf.resolve(config)  # type: ignore
    config = Config(**config)  # type: ignore
    train_ds, val_ds = setup_dataset(config.dataset)
    breakpoint()


if __name__ == "__main__":
    train()
