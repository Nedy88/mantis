"""Training script on TPUs with torch xla."""

import hydra
import torch.distributed as dist
import torch_xla  # type: ignore[reportMissingTypeStubs]
import torch_xla.core.xla_model as xm  # type: ignore[reportMissingTypeStubs]
import torch_xla.distributed.xla_multiprocessing as xmp  # type: ignore[reportMissingTypeStubs]
from omegaconf import OmegaConf

from mantis.configs.config import Config
from mantis.settings import Secrets
from simple_trainer_xla import SimpleTrainerXla


def _mp_fn(index: int, config: Config) -> None:
    dist.init_process_group("xla", init_method="xla://")
    trainer = SimpleTrainerXla(config)
    trainer.train()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(config: Config) -> None:
    """Training script."""
    OmegaConf.resolve(config)  # type: ignore
    config = Config(**config)  # type: ignore
    secrets = Secrets()  # type: ignore[reportUnusedVariable]

    # xmp.spawn(_mp_fn, args=(config,), nprocs=config.world_size)
    torch_xla.launch(_mp_fn, args=(config,))


if __name__ == "__main__":
    train()
