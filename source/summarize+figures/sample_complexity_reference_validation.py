"""Compatibility entry point; implementation: src.experiments.sample_complexity_reference_validation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.sample_complexity_reference_validation import *  # noqa: F401,F403
from src.experiments import sample_complexity_reference_validation as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
