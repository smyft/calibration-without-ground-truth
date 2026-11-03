"""Compatibility entry point for CommonSenseQA thinking_phase1 inference."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from source.LLM_predictions.confidence.logprob.pipeline import (
    choice_letters,
    extract_logprobs_from_metadata as _extract,
    main,
    run_inference,
)


def calculate_logprob_phase1(args):
    return run_inference(args, "thinking_phase1")


if __name__ == "__main__":
    main(dataset_name="CommonSenseQA", mode="thinking_phase1")
