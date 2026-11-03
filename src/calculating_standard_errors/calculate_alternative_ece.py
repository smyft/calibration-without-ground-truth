"""
calculate_alternative_ece.py

Read per-method prediction files saved by experiment_all_baselines.py and compute
six ECE variants (matching experiment_equal_mass_ece.py) for all 12 methods across
3 datasets × 3 models.  Also computes the delta-method standard error and 1.96*SE
(95 % CI half-width) for every ECE variant.

ECE variants
------------
  UWB-10   fixed-width bins, B=10
  UWB-15   fixed-width bins, B=15
  UWB-opt  fixed-width bins, B=floor(n^{1/3})   [Fujisawa & Futami, ICML 2025]
  UMB-10   equal-mass bins,  B=10
  UMB-15   equal-mass bins,  B=15
  UMB-opt  equal-mass bins,  B=floor(n^{1/3})   [Futami & Fujisawa, NeurIPS 2024]

Standard-error formula (delta method, same for UWB and UMB)
-----------------------------------------------------------
  SE = (1/n) * sqrt( sum_m( n_m * Var(d_i in bin m, ddof=1) ) )
  where  d_i = z_i - c_i  (gap between correctness and confidence).
  Bins with fewer than 2 samples contribute 0 to the sum.

Usage
-----
    python3 \\
        src/calculating_standard_errors/calculate_alternative_ece.py \\
        --pred_dir   ./results/all_baselines/predictions \\
        --output_dir ./results/standard_errors
"""

import os
import json
import csv
import argparse

import numpy as np

# Support both module execution and the existing direct-script entry points.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.calculating_standard_errors.common import (
    DATASETS, INST_SLUGS, SHORT_NAME, METHODS, METHOD_CATEGORY,
    load_jsonl, extract_arrays, optimal_n_bins, _uwb_bins, _umb_bins,
    ece_from_bins, se_ece_bins,
)


# ---------------------------------------------------------------------------
# Configuration  (mirrors calculate_standard_errors.py)
# ---------------------------------------------------------------------------


ECE_KEYS = [
    "ece_uwb_10",  "ece_uwb_15",  "ece_uwb_opt",
    "ece_umb_10",  "ece_umb_15",  "ece_umb_opt",
]

CI_SCALE = 1.96   # multiplier to convert SE → 95 % CI half-width


# ---------------------------------------------------------------------------
# ECE point estimates  (identical to experiment_equal_mass_ece.py)
# ---------------------------------------------------------------------------


def ece_uwb(c, z, n_bins):
    return ece_from_bins(c, z, _uwb_bins(c, n_bins))


def ece_umb(c, z, n_bins):
    return ece_from_bins(c, z, _umb_bins(c, n_bins))


# ---------------------------------------------------------------------------
# Delta-method SE for ECE  (same formula for UWB and UMB)
# ---------------------------------------------------------------------------


def se_uwb(c, z, n_bins):
    return se_ece_bins(c, z, _uwb_bins(c, n_bins))


def se_umb(c, z, n_bins):
    return se_ece_bins(c, z, _umb_bins(c, n_bins))


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

                z, c = extract_arrays(records)
                n     = len(records)
                b_opt = optimal_n_bins(n)

                # ---- point estimates ----
                p_uwb_10  = ece_uwb(c, z, 10)
                p_uwb_15  = ece_uwb(c, z, 15)
                p_uwb_opt = ece_uwb(c, z, b_opt)
                p_umb_10  = ece_umb(c, z, 10)
                p_umb_15  = ece_umb(c, z, 15)
                p_umb_opt = ece_umb(c, z, b_opt)

                # ---- standard errors ----
                se_uwb_10  = se_uwb(c, z, 10)
                se_uwb_15  = se_uwb(c, z, 15)
                se_uwb_opt = se_uwb(c, z, b_opt)
                se_umb_10  = se_umb(c, z, 10)
                se_umb_15  = se_umb(c, z, 15)
                se_umb_opt = se_umb(c, z, b_opt)

                row = {
                    "dataset":  dataset,
                    "model":    short,
                    "method":   display,
                    "category": METHOD_CATEGORY.get(display, ""),
                    "n":        n,
                    "b_opt":    b_opt,
                    # point estimates
                    "ece_uwb_10":  round(p_uwb_10,  6),
                    "ece_uwb_15":  round(p_uwb_15,  6),
                    "ece_uwb_opt": round(p_uwb_opt, 6),
                    "ece_umb_10":  round(p_umb_10,  6),
                    "ece_umb_15":  round(p_umb_15,  6),
                    "ece_umb_opt": round(p_umb_opt, 6),
                    # 1.96 * SE  (95 % CI half-width)
                    "ci95_uwb_10":  round(CI_SCALE * se_uwb_10,  6),
                    "ci95_uwb_15":  round(CI_SCALE * se_uwb_15,  6),
                    "ci95_uwb_opt": round(CI_SCALE * se_uwb_opt, 6),
                    "ci95_umb_10":  round(CI_SCALE * se_umb_10,  6),
                    "ci95_umb_15":  round(CI_SCALE * se_umb_15,  6),
                    "ci95_umb_opt": round(CI_SCALE * se_umb_opt, 6),
                }
                rows.append(row)

                print(
                    f"  {dataset:15s} {short:15s} {display:10s}  n={n:5d}  b_opt={b_opt:3d}\n"
                    f"    UWB: 10={p_uwb_10:.4f}±{CI_SCALE*se_uwb_10:.4f}  "
                    f"15={p_uwb_15:.4f}±{CI_SCALE*se_uwb_15:.4f}  "
                    f"opt={p_uwb_opt:.4f}±{CI_SCALE*se_uwb_opt:.4f}\n"
                    f"    UMB: 10={p_umb_10:.4f}±{CI_SCALE*se_umb_10:.4f}  "
                    f"15={p_umb_15:.4f}±{CI_SCALE*se_umb_15:.4f}  "
                    f"opt={p_umb_opt:.4f}±{CI_SCALE*se_umb_opt:.4f}"
                )

    if not rows:
        print("No results found — check --pred_dir.")
        return

    # ---- CSV ----
    fieldnames = [
        "dataset", "model", "method", "category", "n", "b_opt",
        "ece_uwb_10",  "ci95_uwb_10",
        "ece_uwb_15",  "ci95_uwb_15",
        "ece_uwb_opt", "ci95_uwb_opt",
        "ece_umb_10",  "ci95_umb_10",
        "ece_umb_15",  "ci95_umb_15",
        "ece_umb_opt", "ci95_umb_opt",
    ]
    csv_path = os.path.join(output_dir, "alternative_ece.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {csv_path}")

    # ---- JSON ----
    json_path = os.path.join(output_dir, "alternative_ece.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON saved → {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute six ECE variants + 1.96*SE for all calibration methods."
    )
    parser.add_argument("--pred_dir",   default="./results/all_baselines/predictions",
                        help="Root of saved prediction files.")
    parser.add_argument("--output_dir", default="./results/standard_errors",
                        help="Where to write alternative_ece.csv/.json.")
    args = parser.parse_args()
    compute_all(args.pred_dir, args.output_dir)


if __name__ == "__main__":
    main()
