"""Fixture for testing metrics."""
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist

rng = np.random.default_rng(seed=42)

@pytest.fixture(scope="session", autouse=True)
def session_start() -> None:
    """Fixture for session start."""
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    torch.cuda.set_device(0)
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

