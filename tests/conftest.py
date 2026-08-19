"""
Shared pytest fixtures.

The Engine is slow to load (~1s for model loading + dataset), so we use a
session-scoped fixture that loads it once and reuses across all tests.
"""

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path for all test modules
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def engine():
    """Load the Engine once per test session (slow — don't reload per test)."""
    from predict import Engine
    return Engine()


@pytest.fixture(scope="session")
def feature_data():
    """Load the full feature matrix once per test session."""
    from features import build_feature_matrix
    return build_feature_matrix()
