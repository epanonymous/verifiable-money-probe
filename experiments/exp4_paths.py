"""Lightweight shared paths for Wave 2 commands that do not need NumPy."""

from pathlib import Path


EXPERIMENTS_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = EXPERIMENTS_ROOT / "exp3_dataset" / "data"
DEFAULT_LEAK_FREE_DATA = EXPERIMENTS_ROOT / "exp3_dataset" / "data_leak_free"
