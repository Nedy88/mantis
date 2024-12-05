"""Environment setting not uploaded to github."""

from pydantic_settings import BaseSettings


class Secrets(BaseSettings):
    """Holding secrets outside the repo."""

    wandb_api_key: str
    wandb_project: str

    class Config:  # noqa: D106
        env_file = "secrets.env"
