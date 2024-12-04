"""SimpleTrainer for single interation with fixed grid patches."""

import pydantic
from torch import optim

from mantis.configs.config import Config
from mantis.model.simple_model import SimpleModel
from mantis.summary import print_table_with_param_counts


class TrainingState(pydantic.BaseModel):
    """Store the training state for the trainer."""

    epoch: int = 0
    global_step: int = 0
    global_iteration: int = 0
    total_global_steps: int | None = None

    @pydantic.field_validator("epoch")
    @classmethod
    def epoch_must_be_non_negative(cls, v: int) -> int:
        """Validate the epoch."""
        if v < 0:
            msg = f"Epoch must be non-negative. Got {v}."
            raise ValueError(msg)
        return v



class SimpleTrainer:
    """SimpleTrainer for single interation with fixed grid patches."""

    def __init__(self, config: Config, local_rank: int, global_rank: int) -> None:
        self.local_rank = local_rank
        self.config = config
        self.training_state = TrainingState()
        self.model = SimpleModel(config).cuda(self.local_rank)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning.lr,
            weight_decay=config.learning.weight_decay,
        )
        if self.is_global_zero:
            print_table_with_param_counts(self.model, title="Model Parameters")

    def train(self) -> None:
        """Train the model."""

    @property
    def is_global_zero(self) -> bool:
        """Check if the global rank is zero."""
        return self.local_rank == 0

    @property
    def is_step(self) -> bool:
        """Check if the current iteration is a step."""
        return self.training_state.global_iteration % self.config.learning.accum_iter == 0
