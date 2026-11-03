"""Shared arguments for post-processing saved predictions."""

from src.config import DATASET_CONFIG, MODEL_IDS


def add_prediction_arguments(parser):
    """Keep the existing option names and relative results-directory layout."""
    parser.add_argument("--dataset_name", choices=tuple(DATASET_CONFIG), default="MMLU")
    parser.add_argument("--split", default="test")
    parser.add_argument("--confidence_dir", default="./results")
    for prefix, default in (
        ("model1", "Qwen/Qwen3-8B"),
        ("model2", "Qwen/Qwen3-8B-Base"),
    ):
        parser.add_argument(f"--{prefix}_type", choices=("LLMs",), default="LLMs")
        parser.add_argument(f"--{prefix}_method", choices=("logprob",), default="logprob")
        parser.add_argument(f"--{prefix}_name", choices=MODEL_IDS, default=default)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--debug", action="store_true")
    return parser
