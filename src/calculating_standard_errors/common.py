"""Shared saved-prediction readers and evaluation conventions.

Keep bin boundaries, row order, method names, and historical table labels stable.
"""

import json
import math
import os

import numpy as np

from src.config import DATASETS, MODEL_PAIRS, SHORT_NAME as MODEL_SHORT_NAME

INST_SLUGS = [inst for inst, _ in MODEL_PAIRS]
# The standard-error tables historically spell out the Ministral generation.
SHORT_NAME = {
    slug: name.replace("Ministral-8B", "Ministral-3-8B")
    for slug, name in MODEL_SHORT_NAME.items()
}

METHODS = [
    ("Instruct",               "Instruct"),
    ("Base",                   "Base"),
    ("Ours",                   "Ours"),
    ("BCT",                    "BCT"),
    ("ConfAvg",                "ConfAvg"),
    ("DACA",                   "DACA"),
    ("Ours_ShuffledRandomRef", "DummyRef"),
    ("TS",                     "TS"),
    ("HB",                     "HB"),
    ("IR",                     "IR"),
    ("GPcalib",                "GPcalib"),
    ("PBR",                    "PBR"),
]

METHOD_CATEGORY = {
    "Instruct": "uncalibrated",
    "Base":     "reference",
    "Ours":     "label-free",
    "BCT":      "label-free",
    "ConfAvg":  "label-free",
    "DACA":     "label-free",
    "DummyRef": "label-free",
    "TS":       "supervised",
    "HB":       "supervised",
    "IR":       "supervised",
    "GPcalib":  "supervised",
    "PBR":      "supervised",
}


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


def get_z(records):
    """Correctness indicator z_i ∈ {0, 1}."""
    return np.array(
        [float(r["choices"][r["gold"]] == r["model_prediction"]["final_answer"])
         for r in records]
    )


def get_c(records):
    """Calibrated confidence ĉ_i ∈ [0, 1]."""
    return np.array([float(r["model_prediction"]["confidence"]) for r in records])


def extract_arrays(records):
    """Return correctness and confidence arrays without changing record order."""
    return get_z(records), get_c(records)


def optimal_n_bins(n):
    return max(1, int(np.floor(n ** (1.0 / 3.0))))


def _uwb_bins(c, n_bins):
    """Yield index arrays for each UWB bin."""
    limits = np.linspace(0.0, 1.0, n_bins + 1)
    for i, (lo, hi) in enumerate(zip(limits[:-1], limits[1:])):
        mask = (c <= hi) if i == 0 else ((c > lo) & (c <= hi))
        idx = np.where(mask)[0]
        if len(idx) > 0:
            yield idx


def _umb_bins(c, n_bins):
    """Yield index arrays for each UMB (equal-mass) bin."""
    n = len(c)
    order = np.argsort(c)
    for idx_local in np.array_split(np.arange(n), n_bins):
        if len(idx_local) > 0:
            yield order[idx_local]


def ece_from_bins(c, z, bin_gen):
    """Compute ECE from a bin generator that yields index arrays."""
    n = len(c)
    ece = 0.0
    for idx in bin_gen:
        ece += abs(c[idx].mean() - z[idx].mean()) * len(idx) / n
    return float(ece)


def se_ece_bins(c, z, bin_gen):
    """Delta-method SE using the supplied bin-index generator.

    SE = (1/n) * sqrt( sum_m( n_m * Var(d_i, ddof=1) ) )
    where d_i = z_i - c_i.  Bins with < 2 samples are skipped.
    """
    d = z - c
    n = len(c)
    total = 0.0
    for idx in bin_gen:
        n_m = len(idx)
        if n_m < 2:
            continue
        total += n_m * float(np.var(d[idx], ddof=1))
    return float(math.sqrt(total) / n)
