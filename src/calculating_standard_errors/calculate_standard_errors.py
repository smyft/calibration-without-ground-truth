"""
calculate_standard_errors.py

Compute closed-form standard errors for all four evaluation metrics
(Accuracy, Brier Score, Confidence Loss, ECE UWB B=10) from saved
per-method prediction files produced by experiment_all_baselines.py.

Formulas follow:
  src/calculating_standard_errors/standard_errors_summary.md

Methods reported:
  Instruct, Base, Ours, BCT, ConfAvg, DACA,
  DummyRef (= Ours_ShuffledRandomRef),
  TS, HB, IR, GPcalib, PBR

Datasets:  MMLU, MMLU_Redux, CommonSenseQA
Models:    Qwen3-8B, LLaMA-3.1-8B, Ministral-3-8B

Usage:
    python3 \\
        src/calculating_standard_errors/calculate_standard_errors.py \\
        --pred_dir  ./results/all_baselines/predictions \\
        --output_dir ./results/standard_errors
"""

import os
import json
import csv
import math
import argparse

import numpy as np

# Support both module execution and the existing direct-script entry points.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.calculating_standard_errors.common import (
    DATASETS, INST_SLUGS, SHORT_NAME, METHODS, METHOD_CATEGORY,
    load_jsonl, get_z, get_c, _uwb_bins, ece_from_bins, se_ece_bins,
)


ECE_N_BINS = 10


# ---------------------------------------------------------------------------
# Point estimates  (match experiment_all_baselines.py exactly)
# ---------------------------------------------------------------------------

def point_accuracy(z):
    return float(np.mean(z))


def point_brier(c, z):
    return float(np.mean((c - z) ** 2))


def point_cl(c, z):
    return point_brier(c, z) - point_accuracy(z)


def point_ece_uwb(c, z, n_bins=ECE_N_BINS):
    """Equal-width ECE; the first bin includes confidence zero."""
    return ece_from_bins(c, z, _uwb_bins(c, n_bins))


# ---------------------------------------------------------------------------
# Standard error formulas  (standard_errors_summary.md)
# ---------------------------------------------------------------------------

def se_accuracy(z):
    """SE = sqrt(acc * (1 - acc) / n)  — binomial formula."""
    n   = len(z)
    acc = point_accuracy(z)
    return math.sqrt(acc * (1.0 - acc) / n)


def se_brier(c, z):
    """SE = std(b_i, ddof=1) / sqrt(n),  b_i = (ĉ_i - z_i)^2."""
    b = (c - z) ** 2
    return float(np.std(b, ddof=1) / math.sqrt(len(b)))


def se_cl(c, z):
    """SE = std(ℓ_i, ddof=1) / sqrt(n),  ℓ_i = b_i - z_i.
    Computed from per-example values to handle the BS/Acc correlation.
    """
    b = (c - z) ** 2
    l = b - z
    return float(np.std(l, ddof=1) / math.sqrt(len(l)))


def se_ece_uwb(c, z, n_bins=ECE_N_BINS):
    """Delta-method standard error, skipping bins with fewer than two samples."""
    return se_ece_bins(c, z, _uwb_bins(c, n_bins))


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_all(pred_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        for inst_slug in INST_SLUGS:
            short = SHORT_NAME[inst_slug]
            for file_stem, display in METHODS:
                pred_path = os.path.join(pred_dir, dataset, inst_slug,
                                         f"{file_stem}.jsonl")
                records = load_jsonl(pred_path)
                if not records:
                    print(f"  [SKIP] {dataset}/{short}/{display} — not found: {pred_path}")
                    continue

                z   = get_z(records)
                c   = get_c(records)
                n   = len(records)

                acc  = point_accuracy(z)
                bs   = point_brier(c, z)
                cl   = point_cl(c, z)
                ece  = point_ece_uwb(c, z)

                se_acc_v  = se_accuracy(z)
                se_bs_v   = se_brier(c, z)
                se_cl_v   = se_cl(c, z)
                se_ece_v  = se_ece_uwb(c, z)

                rows.append({
                    "dataset":   dataset,
                    "model":     short,
                    "method":    display,
                    "category":  METHOD_CATEGORY.get(display, ""),
                    "n":         n,
                    "acc":       round(acc,       6),
                    "se_acc":    round(se_acc_v,  6),
                    "brier":     round(bs,        6),
                    "se_brier":  round(se_bs_v,   6),
                    "cl":        round(cl,        6),
                    "se_cl":     round(se_cl_v,   6),
                    "ece":       round(ece,       6),
                    "se_ece":    round(se_ece_v,  6),
                })
                print(f"  {dataset:15s} {short:15s} {display:10s}  n={n:5d}  "
                      f"acc={acc:.4f}±{se_acc_v:.4f}  "
                      f"brier={bs:.4f}±{se_bs_v:.4f}  "
                      f"ece={ece:.4f}±{se_ece_v:.4f}  "
                      f"cl={cl:.4f}±{se_cl_v:.4f}")

    if not rows:
        print("No results — check --pred_dir.")
        return

    # Save CSV
    fieldnames = ["dataset", "model", "method", "category", "n",
                  "acc", "se_acc", "brier", "se_brier",
                  "cl", "se_cl", "ece", "se_ece"]
    csv_path = os.path.join(output_dir, "standard_errors.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "standard_errors.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON saved → {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute closed-form SEs for all calibration methods."
    )
    parser.add_argument("--pred_dir",    default="./results/all_baselines/predictions",
                        help="Root of saved prediction files.")
    parser.add_argument("--output_dir",  default="./results/standard_errors",
                        help="Where to write standard_errors.csv/.json.")
    args = parser.parse_args()
    compute_all(args.pred_dir, args.output_dir)


if __name__ == "__main__":
    main()
