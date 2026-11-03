"""Compatibility entry point; implementation: src.experiments.experiment_precision_ablation."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.experiment_precision_ablation import *  # noqa: F401,F403
from src.experiments import experiment_precision_ablation as _implementation


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    _implementation.main()
