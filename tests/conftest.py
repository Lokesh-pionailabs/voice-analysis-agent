"""Shared pytest fixtures."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure required settings exist before any module imports `get_settings()`.
os.environ.setdefault("GCP_PROJECT_ID", "test-project")
os.environ.setdefault("GCS_BUCKET_NAME", "test-bucket")
os.environ.setdefault("APP_ENV", "test")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure `get_settings()` re-reads env vars fresh for each test."""
    from app.utils.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
