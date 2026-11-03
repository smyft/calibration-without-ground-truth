"""Supported model pairs, local data/checkpoints, and evaluation splits.

Model identifiers determine result paths; local directory names only determine
where weights are read. Resolving a checkpoint never contacts a model hub.

Set ``MODEL_ROOT`` and ``DATASET_ROOT`` in the process environment for custom
storage locations. Defaults are ``models/`` and ``datasets/`` alongside the
project containing this code repository; no machine-specific path is required
in source control.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = CODE_ROOT.parent.parent / "models"
DEFAULT_DATASET_ROOT = CODE_ROOT.parent.parent / "datasets"
LOCAL_DATASET_DIRS = {
    "MMLU": "MMLU",
    "MMLU_Redux": "MMLU-Redux-2.0",
    "CommonSenseQA": "CommonSenseQA",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    variant: str
    local_dir: str
    display_name: str

    @property
    def slug(self):
        return self.model_id.replace("/", "_")


_SPECS = (
    ModelSpec("Qwen/Qwen3-8B", "qwen", "instruct", "Qwen3-8B", "Qwen3-8B"),
    ModelSpec("Qwen/Qwen3-8B-Base", "qwen", "base", "Qwen3-8B-Base", "Qwen3-8B"),
    ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama", "instruct",
              "Llama-3.1-8B-Instruct", "LLaMA-3.1-8B"),
    ModelSpec("meta-llama/Llama-3.1-8B", "llama", "base",
              "Llama-3.1-8B", "LLaMA-3.1-8B"),
    ModelSpec("mistralai/Ministral-3-8B-Instruct-2512", "ministral", "instruct",
              "Ministral-3-8B-Instruct-2512-BF16", "Ministral-8B"),
    ModelSpec("mistralai/Ministral-3-8B-Base-2512", "ministral", "base",
              "Ministral-3-8B-Base-2512", "Ministral-8B"),
)
MODEL_SPECS = {spec.model_id: spec for spec in _SPECS}
MODEL_IDS = tuple(MODEL_SPECS)
MODEL_SLUGS = tuple(spec.slug for spec in _SPECS)
MODEL_PAIRS = [(inst.slug, base.slug) for inst, base in zip(_SPECS[::2], _SPECS[1::2])]
SHORT_NAME = {spec.slug: spec.display_name for spec in _SPECS if spec.variant == "instruct"}

DATASET_CONFIG = {
    "MMLU": {
        "calib_dataset": "MMLU", "calib_split": "validation",
        "apply_dataset": "MMLU", "apply_split": "test",
        "inst_tpl": "baseline_thinking", "instruct_tpl": "baseline_thinking",
        "base_tpl": "baseline", "gp_max_samples": None,
    },
    "MMLU_Redux": {
        "calib_dataset": "MMLU", "calib_split": "validation",
        "apply_dataset": "MMLU_Redux", "apply_split": "test",
        "inst_tpl": "baseline_thinking", "instruct_tpl": "baseline_thinking",
        "base_tpl": "baseline", "gp_max_samples": None,
    },
    "CommonSenseQA": {
        "calib_dataset": "CommonSenseQA", "calib_split": "train",
        "apply_dataset": "CommonSenseQA", "apply_split": "validation",
        "inst_tpl": "baseline_thinking", "instruct_tpl": "baseline_thinking",
        "base_tpl": "baseline", "gp_max_samples": 1500,
    },
}
DATASETS = tuple(DATASET_CONFIG)
MAIN_PAIRS = [(dataset, inst, base) for dataset in DATASETS for inst, base in MODEL_PAIRS]


def get_model_spec(name):
    """Resolve a supported hub ID, result slug, or checkpoint directory name."""
    for spec in _SPECS:
        if name in (spec.model_id, spec.slug, spec.local_dir):
            return spec
    raise ValueError(f"Unsupported model {name!r}. Choose one of: {', '.join(MODEL_IDS)}")


def resolve_model_path(name, model_root=None, model_path=None):
    """Find a complete local checkpoint, failing before inference if missing.

    ``model_path`` overrides ``model_root``, which overrides ``MODEL_ROOT`` in
    the environment. Without an override, ``DEFAULT_MODEL_ROOT`` is used. The
    canonical model ID is still used for result filenames, including when an
    explicit checkpoint path is used.
    """
    spec = get_model_spec(name)
    root = Path(model_root or os.environ.get("MODEL_ROOT", DEFAULT_MODEL_ROOT)).expanduser()
    path = Path(model_path).expanduser() if model_path else root / spec.local_dir
    path = path.resolve()
    required = [path / "config.json", path / "tokenizer_config.json"]
    if spec.family == "ministral":
        required.extend([path / "params.json", path / "tekken.json"])
    missing = [str(p) for p in required if not p.is_file()]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        with index.open(encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Empty checkpoint weight index: {index}")
        missing.extend(str(path / shard) for shard in sorted(set(weight_map.values()))
                       if not (path / shard).is_file())
    elif not (path / "model.safetensors").is_file():
        missing.append(str(path / "model.safetensors.index.json"))
    if not any((path / candidate).is_file() for candidate in ("tokenizer.json", "tekken.json")):
        missing.append(str(path / "tokenizer.json"))
    if missing:
        raise FileNotFoundError(
            f"Local checkpoint for {spec.model_id} is incomplete at {path}. "
            "No download was attempted. Missing: " + ", ".join(missing)
        )
    return path
