"""Fixtures for testing."""
import numpy as np
import pytest

rng = np.random.default_rng(seed=42)


@pytest.fixture(params=list(rng.uniform(0.1, 0.9, 20)))
def drop_path_prob(request: pytest.FixtureRequest) -> float:
    """Drop probability for DropPath module testing."""
    return request.param
