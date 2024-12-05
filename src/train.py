"""Training script."""
import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from mantis.configs.config import Config
from simple_trainer import SimpleTrainer


def ddp_setup(backend: str) -> tuple[int, int, int]:
    """DDP setup."""
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend=backend)

    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(config: Config) -> None:
    """Training script."""
    OmegaConf.resolve(config)  # type: ignore
    config = Config(**config)  # type: ignore
    torch.set_float32_matmul_precision(config.float32_matmul_precision)

    local_rank, global_rank, world_size = ddp_setup("gloo")
    assert world_size == config.num_nodes * config.gpus_per_node

    trainer = SimpleTrainer(config, local_rank, global_rank)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
