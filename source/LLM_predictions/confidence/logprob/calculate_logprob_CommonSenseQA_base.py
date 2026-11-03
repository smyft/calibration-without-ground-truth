"""Compatibility entry point for CommonSenseQA base inference."""

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


def extract_logprobs_from_metadata(top_logprobs):
    return _extract(top_logprobs, choice_letters("CommonSenseQA", "base"))


def calculate_logprob(args):
    return run_inference(args, "base")


if __name__ == "__main__":
    main(dataset_name="CommonSenseQA", mode="base")
