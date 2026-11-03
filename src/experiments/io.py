"""Shared prediction I/O and the fixed calibration/evaluation split protocol."""

import json
import os
from src.config import DATASET_CONFIG

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def save_jsonl(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

def _conf_path(results_dir, dataset, split, slug, tpl):
    return os.path.join(results_dir, dataset, split,
                        "LLMs", "confidence", "logprob", slug, f"{tpl}.jsonl")

def _ours_output_path(results_dir, apply_dataset, apply_split,
                      calib_dataset, calib_split,
                      model1_slug, model1_tpl, ref_slug, ref_tpl):
    return os.path.join(
        results_dir, apply_dataset, apply_split,
        "post_processing", "transformation", "confidence",
        f"calibrated_on_{calib_dataset}_{calib_split}",
        "confidence", "logprob", model1_slug,
        f"{model1_tpl}_{ref_slug}_{ref_tpl}.jsonl",
    )


def load_pair_data(results_dir, dataset, inst_slug, base_slug):
    """Return calibration instruct/base and held-out instruct/base, in file order."""
    cfg = DATASET_CONFIG[dataset]
    return tuple(
        load_jsonl(_conf_path(results_dir, cfg[f"{split}_dataset"],
                             cfg[f"{split}_split"], slug, template))
        for split in ("calib", "apply")
        for slug, template in ((inst_slug, cfg["inst_tpl"]),
                               (base_slug, cfg["base_tpl"]))
    )
