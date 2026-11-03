"""
experiment_supervised_baselines_at_nstar.py

For each (dataset, model pair), samples N* labeled examples from the
calibration set using a reference-validation label budget, then fits
Temperature Scaling (TS), Histogram
Binning (HB), and Isotonic Regression (IR) on those N* samples and
evaluates on the held-out test set.

This directly answers: "How well do supervised baselines perform when
given only the label budget required to validate the reference assumption?"

Historical fixed budgets retained for reproducibility (the default mode):
  Qwen3-8B,      MMLU-Redux:      N* = 170   (full calib = 1,531)
  LLaMA-3.1-8B,  MMLU-Redux:      N* =  50   (full calib = 1,531)
  Ministral-3-8B,MMLU-Redux:      N* = 130   (full calib = 1,531)
  Qwen3-8B,      CommonSenseQA:   N* = 190   (full calib = 9,741)
  LLaMA-3.1-8B,  CommonSenseQA:   N* =  80   (full calib = 9,741)
  Ministral-3-8B,CommonSenseQA:   N* = 130   (full calib = 9,741)

Each N*-fitted result is averaged over n_reps random subsamples to reduce
sampling variance, matching the approach in run_all_sample_complexity.py.
The full-calibration-set baselines (standard paper results) are computed
once as a reference.

Usage:
    python \\
        source/summarize+figures/experiment_supervised_baselines_at_nstar.py \\
        --results_dir ./results \\
        --output_dir  ./results/supervised_at_nstar \\
        --n_reps      200 \\
        --seed        42
"""

from src.experiments.cli import add_selection_arguments, selected_pairs

from src.experiments.io import _conf_path, load_jsonl
from src.experiments.settings import SHORT_NAME
from src.config import DATASET_CONFIG
from src.experiments.budget_baselines import fit_apply_hb, fit_apply_ir, fit_apply_ts
from src.metrics import is_correct
from src.config import MODEL_PAIRS
BASE_SLUG = dict(MODEL_PAIRS)

import os
import json
import argparse
import csv

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Historical fixed label budgets per (dataset, instruct_slug).
N_STAR = {
    ("MMLU_Redux",    "Qwen_Qwen3-8B"):                          170,
    ("MMLU_Redux",    "meta-llama_Llama-3.1-8B-Instruct"):        50,
    ("MMLU_Redux",    "mistralai_Ministral-3-8B-Instruct-2512"): 130,
    ("CommonSenseQA", "Qwen_Qwen3-8B"):                          190,
    ("CommonSenseQA", "meta-llama_Llama-3.1-8B-Instruct"):        80,
    ("CommonSenseQA", "mistralai_Ministral-3-8B-Instruct-2512"): 130,
}

# Dataset configuration: calibration source and evaluation target


# Ordered pairs for consistent output
PAIRS = list(N_STAR)

# Path to pre-computed "Ours" results (Bregman projection, full calibration)
def _ours_path(results_dir, dataset, inst_slug, base_slug, calib_dataset, calib_split, inst_tpl):
    pp_file = f"{inst_tpl}_{base_slug}_baseline.jsonl"
    return os.path.join(
        results_dir, dataset,
        DATASET_CONFIG[dataset]["apply_split"],
        "post_processing", "transformation", "confidence",
        f"calibrated_on_{calib_dataset}_{calib_split}",
        "confidence", "logprob", inst_slug, pp_file,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def ece_uwb(records, n_bins=10):
    confs   = np.array([float(r["model_prediction"]["confidence"]) for r in records])
    correct = np.array([float(is_correct(r)) for r in records])
    if len(confs) == 0:
        return float("nan")
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bin_limits[:-1], bin_limits[1:])):
        mask = (confs <= hi) if i == 0 else ((confs > lo) & (confs <= hi))
        if mask.sum() == 0:
            continue
        ece += abs(confs[mask].mean() - correct[mask].mean()) * mask.sum() / len(confs)
    return float(ece)


def optimal_n_bins(n: int) -> int:
    """B* = floor(n^{1/3}) per Futami & Fujisawa (NeurIPS 2024) and
    Fujisawa & Futami (ICML 2025). Minimises total ECE estimation bias."""
    return max(1, int(np.floor(n ** (1.0 / 3.0))))


def calculate_ece_umb(is_correct_arr: np.ndarray,
                      confidences: np.ndarray,
                      n_bins: int) -> float:
    """Equal-mass (UMB / AdaptiveECE) ECE. Each bin gets an equal number of
    samples (sorted by confidence); the last bin absorbs any remainder."""
    is_correct_arr = np.asarray(is_correct_arr, dtype=float)
    confidences    = np.asarray(confidences,    dtype=float)
    n = len(confidences)
    if n == 0:
        return float("nan")
    order     = np.argsort(confidences)
    s_conf    = confidences[order]
    s_correct = is_correct_arr[order]
    ece = 0.0
    for idx in np.array_split(np.arange(n), n_bins):
        if len(idx) == 0:
            continue
        ece += abs(s_conf[idx].mean() - s_correct[idx].mean()) * len(idx) / n
    return float(ece)


def ece_umb_opt(records) -> float:
    """UMB ECE with optimal bin count B* = floor(n^{1/3}) applied to a list
    of prediction records. This is the primary ECE metric used in this script."""
    confs   = np.array([float(r["model_prediction"]["confidence"]) for r in records])
    correct = np.array([float(is_correct(r)) for r in records])
    if len(confs) == 0:
        return float("nan")
    return calculate_ece_umb(correct, confs, n_bins=optimal_n_bins(len(confs)))


# ---------------------------------------------------------------------------
# Supervised baseline implementations
# (using the modular class architectures mapping confidence -> P(correct))
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Supervised baseline implementations
# (scalar mode: fit on (confidence, is_correct) pairs from calib_records,
#  apply to test_records preserving final_answer)
# ---------------------------------------------------------------------------

# Core per-pair experiment
# ---------------------------------------------------------------------------

def run_pair(dataset, inst_slug, results_dir, n_reps, seed):
    """
    Returns a result dict for one (dataset, inst_slug) pair, or None on skip.

    Computes:
      - full-calibration TS / HB / IR ECE (once, on all calib records)
      - N*-bootstrap TS / HB / IR ECE (mean ± 95% CI over n_reps bootstrap draws)
      - Ours ECE (loaded from pre-computed post-processing output)
      - Instruct ECE (uncalibrated)
      - Per-bootstrap improvement of Ours over each supervised baseline at N*:
          improvement_X = ECE(X@N*_rep) - ECE(Ours_full)
        A positive value means the supervised baseline at N* is worse than Ours
        (label-free), so higher improvement = bigger advantage for our method.
        95% CI is the [2.5, 97.5] percentile over the n_reps bootstrap samples.

    Bootstrap protocol: each of the n_reps draws samples N* calibration records
    *with replacement* from the full calibration set (standard non-parametric
    bootstrap), matching the analysis in run_all_sample_complexity.py.
    """
    cfg        = DATASET_CONFIG[dataset]
    calib_ds   = cfg["calib_dataset"]
    calib_sp   = cfg["calib_split"]
    apply_ds   = cfg["apply_dataset"]
    apply_sp   = cfg["apply_split"]
    inst_tpl   = cfg["instruct_tpl"]
    n_star     = N_STAR[(dataset, inst_slug)]
    base_slug  = BASE_SLUG[inst_slug]
    short      = SHORT_NAME.get(inst_slug, inst_slug)

    # Load calibration and test files
    calib_path = _conf_path(results_dir, calib_ds, calib_sp, inst_slug, inst_tpl)
    test_path  = _conf_path(results_dir, apply_ds, apply_sp, inst_slug, inst_tpl)
    calib_recs = load_jsonl(calib_path)
    test_recs  = load_jsonl(test_path)

    if not calib_recs:
        print(f"  [SKIP] Missing calib: {calib_path}")
        return None
    if not test_recs:
        print(f"  [SKIP] Missing test: {test_path}")
        return None

    n_calib = len(calib_recs)
    n_test  = len(test_recs)

    if n_star > n_calib:
        print(f"  [WARN] N*={n_star} > n_calib={n_calib}; clamping to n_calib.")
        n_star = n_calib

    # ---- Instruct (uncalibrated) ECE on test ----
    ece_inst = ece_umb_opt(test_recs)

    # ---- Full-calibration baselines (reference, fitted once) ----
    print(f"  Full-calibration baselines (n={n_calib})...")
    ts_full, T_full  = fit_apply_ts(calib_recs, test_recs)
    hb_full          = fit_apply_hb(calib_recs, test_recs)
    ir_full          = fit_apply_ir(calib_recs, test_recs)
    ece_ts_full      = ece_umb_opt(ts_full)
    ece_hb_full      = ece_umb_opt(hb_full)
    ece_ir_full      = ece_umb_opt(ir_full)

    # ---- Load pre-computed "Ours" ECE ----
    ours_path = _ours_path(results_dir, apply_ds, inst_slug, base_slug,
                           calib_ds, calib_sp, inst_tpl)
    ours_recs = load_jsonl(ours_path)
    ece_ours  = ece_umb_opt(ours_recs) if ours_recs else float("nan")

    # ---- N*-bootstrap baselines (n_reps bootstrap draws with replacement) ----
    print(f"  N*={n_star} bootstrap baselines ({n_reps} reps, with replacement)...")
    ts_eces, hb_eces, ir_eces = [], [], []
    T_vals = []

    for rep in range(n_reps):
        rng = np.random.default_rng(seed * 10000 + rep)
        idx = rng.choice(n_calib, size=n_star, replace=True)   # bootstrap resample
        sub = [calib_recs[i] for i in idx]

        ts_sub, T_sub = fit_apply_ts(sub, test_recs)
        hb_sub        = fit_apply_hb(sub, test_recs)
        ir_sub        = fit_apply_ir(sub, test_recs)

        ts_eces.append(ece_umb_opt(ts_sub))
        hb_eces.append(ece_umb_opt(hb_sub))
        ir_eces.append(ece_umb_opt(ir_sub))
        T_vals.append(T_sub)

    ts_eces = np.array(ts_eces)
    hb_eces = np.array(hb_eces)
    ir_eces = np.array(ir_eces)

    # improvement_X[rep] = ECE(X@N*_rep) - ECE(Ours_full)
    # Positive means supervised baseline at N* is worse (higher ECE) than Ours,
    # i.e., our label-free method has that much ECE advantage.
    imp_ts = ts_eces - ece_ours
    imp_hb = hb_eces - ece_ours
    imp_ir = ir_eces - ece_ours

    def _ci(arr):
        """Return (mean, ci_lo, ci_hi) — 95% bootstrap percentile CI."""
        return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    ts_mean, ts_ci_lo, ts_ci_hi = _ci(ts_eces)
    hb_mean, hb_ci_lo, hb_ci_hi = _ci(hb_eces)
    ir_mean, ir_ci_lo, ir_ci_hi = _ci(ir_eces)

    imp_ts_mean, imp_ts_ci_lo, imp_ts_ci_hi = _ci(imp_ts)
    imp_hb_mean, imp_hb_ci_lo, imp_hb_ci_hi = _ci(imp_hb)
    imp_ir_mean, imp_ir_ci_lo, imp_ir_ci_hi = _ci(imp_ir)

    return {
        "dataset":    dataset,
        "inst_slug":  inst_slug,
        "base_slug":  base_slug,
        "short_name": short,
        "n_star":     n_star,
        "n_calib":    n_calib,
        "n_test":     n_test,
        "n_reps":     n_reps,
        # Uncalibrated Instruct (test set)
        "ece_inst":   ece_inst,
        # Full-calibration baselines (test set)
        "ece_ts_full":  ece_ts_full,
        "ece_hb_full":  ece_hb_full,
        "ece_ir_full":  ece_ir_full,
        "T_full":       T_full,
        # Ours (pre-computed, label-free, full calibration set, test set)
        "ece_ours_full": ece_ours,
        # N*-bootstrap baselines: mean and 95% CI over n_reps bootstrap draws
        "ece_ts_nstar_mean":  ts_mean,
        "ece_ts_nstar_ci_lo": ts_ci_lo,
        "ece_ts_nstar_ci_hi": ts_ci_hi,
        "ece_hb_nstar_mean":  hb_mean,
        "ece_hb_nstar_ci_lo": hb_ci_lo,
        "ece_hb_nstar_ci_hi": hb_ci_hi,
        "ece_ir_nstar_mean":  ir_mean,
        "ece_ir_nstar_ci_lo": ir_ci_lo,
        "ece_ir_nstar_ci_hi": ir_ci_hi,
        "T_nstar_mean": float(np.mean(T_vals)),
        "T_nstar_std":  float(np.std(T_vals)),
        # Improvement of Ours over each supervised baseline at N*:
        #   improvement = ECE(supervised@N*_rep) - ECE(Ours_full)
        #   positive = Ours is better (lower ECE); 95% CI across bootstrap draws
        "improvement_ts_mean":  imp_ts_mean,
        "improvement_ts_ci_lo": imp_ts_ci_lo,
        "improvement_ts_ci_hi": imp_ts_ci_hi,
        "improvement_hb_mean":  imp_hb_mean,
        "improvement_hb_ci_lo": imp_hb_ci_lo,
        "improvement_hb_ci_hi": imp_hb_ci_hi,
        "improvement_ir_mean":  imp_ir_mean,
        "improvement_ir_ci_lo": imp_ir_ci_lo,
        "improvement_ir_ci_hi": imp_ir_ci_hi,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "dataset", "short_name", "n_star", "n_calib", "n_test", "n_reps",
    "ece_inst",
    "ece_ours_full",
    "ece_ts_full", "ece_hb_full", "ece_ir_full",
    # N*-bootstrap ECE (mean and 95% CI)
    "ece_ts_nstar_mean", "ece_ts_nstar_ci_lo", "ece_ts_nstar_ci_hi",
    "ece_hb_nstar_mean", "ece_hb_nstar_ci_lo", "ece_hb_nstar_ci_hi",
    "ece_ir_nstar_mean", "ece_ir_nstar_ci_lo", "ece_ir_nstar_ci_hi",
    # Improvement = ECE(supervised@N*) - ECE(Ours_full); positive → Ours wins
    "improvement_ts_mean", "improvement_ts_ci_lo", "improvement_ts_ci_hi",
    "improvement_hb_mean", "improvement_hb_ci_lo", "improvement_hb_ci_hi",
    "improvement_ir_mean", "improvement_ir_ci_lo", "improvement_ir_ci_hi",
]


def print_summary(rows):
    """Print a human-readable summary table to stdout."""
    W = 140
    n_reps = rows[0]["n_reps"] if rows else "?"
    print(f"\n{'='*W}")
    print(f"Supervised Baselines at N* (bootstrap, with replacement) — ECE (UMB-opt, B*=⌊n^{{1/3}}⌋) on test set")
    print(f"N* = min labeled samples to confirm Base ECE < Instruct ECE (95% CI)")
    print(f"Improvement = ECE(supervised@N*) − ECE(Ours_full)  [positive → Ours wins]")
    print(f"95% CI across {n_reps} bootstrap draws shown as [lo, hi]")
    print(f"{'='*W}")
    print(
        f"{'Dataset':<15} {'Model':<18} {'N*':>5} "
        f"{'Inst':>7} {'Ours':>7} "
        f"{'TS_full':>8} {'HB_full':>8} {'IR_full':>8}  │  "
        f"{'TS@N*(mean)':>12} {'95%CI':>15}  "
        f"{'Imp_TS(mean)':>13} {'95%CI':>15}"
    )
    print(f"{'-'*W}")

    def _f(v):
        return f"{v:.4f}" if (v is not None and not (isinstance(v, float) and np.isnan(v))) else "   N/A"

    def _ci(lo, hi):
        if any(isinstance(x, float) and np.isnan(x) for x in [lo, hi]):
            return "      N/A"
        return f"[{lo:.4f},{hi:.4f}]"

    prev_ds = None
    for r in rows:
        if r["dataset"] != prev_ds and prev_ds is not None:
            print(f"{'-'*W}")
        prev_ds = r["dataset"]
        print(
            f"{r['dataset']:<15} {r['short_name']:<18} {r['n_star']:>5} "
            f"{_f(r['ece_inst']):>7} {_f(r['ece_ours_full']):>7} "
            f"{_f(r['ece_ts_full']):>8} {_f(r['ece_hb_full']):>8} {_f(r['ece_ir_full']):>8}  │  "
            f"{_f(r['ece_ts_nstar_mean']):>12} {_ci(r['ece_ts_nstar_ci_lo'], r['ece_ts_nstar_ci_hi']):>15}  "
            f"{_f(r['improvement_ts_mean']):>13} {_ci(r['improvement_ts_ci_lo'], r['improvement_ts_ci_hi']):>15}"
        )
    print(f"{'='*W}")
    print(
        "Columns:\n"
        "  Inst      : uncalibrated instruct ECE\n"
        "  Ours      : Bregman projection (label-free, full calibration set)\n"
        "  TS/HB/IR _full : supervised baselines fitted on full calibration set\n"
        f"  TS@N*(mean)    : TS ECE (UMB-opt) fitted on N* bootstrap samples (mean, {n_reps} draws)\n"
        "  Imp_TS    : ECE(TS@N*) − ECE(Ours_full); positive = Ours is better\n"
        "  (Full improvement table saved in results JSON/CSV)\n"
    )


def save_outputs(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"JSON saved → {json_path}")

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved  → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Supervised baselines at N* label budget vs full calibration."
    )
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir",  default="./results/supervised_at_nstar")
    parser.add_argument("--n_reps",      type=int, default=200,
                        help="Number of random subsamples per N* (default: 200).")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--fast",        action="store_true",
                        help="Quick smoke-test: n_reps=20.")
    add_selection_arguments(parser, datasets=("MMLU_Redux", "CommonSenseQA"))
    args = parser.parse_args(argv)

    if args.fast:
        args.n_reps = 20

    all_rows = []

    for dataset, inst_slug in selected_pairs(PAIRS, args):
        short = SHORT_NAME.get(inst_slug, inst_slug)
        n_star = N_STAR[(dataset, inst_slug)]
        print(f"\n{'='*60}")
        print(f"  {dataset} | {short}  (N*={n_star})")
        print(f"{'='*60}")

        row = run_pair(dataset, inst_slug, args.results_dir, args.n_reps, args.seed)
        if row is None:
            continue
        all_rows.append(row)

        def _ci_str(lo, hi): return f"[{lo:.4f}, {hi:.4f}]"
        print(f"  Instruct ECE:              {row['ece_inst']:.4f}")
        print(f"  Ours (full calib) ECE:     {row['ece_ours_full']:.4f}")
        print(f"  TS (full calib)   ECE:     {row['ece_ts_full']:.4f}")
        print(f"  TS  @ N*={n_star:<4d}  ECE: {row['ece_ts_nstar_mean']:.4f}  "
              f"95%CI {_ci_str(row['ece_ts_nstar_ci_lo'], row['ece_ts_nstar_ci_hi'])}  "
              f"Imp {row['improvement_ts_mean']:+.4f} {_ci_str(row['improvement_ts_ci_lo'], row['improvement_ts_ci_hi'])}")
        print(f"  HB  @ N*={n_star:<4d}  ECE: {row['ece_hb_nstar_mean']:.4f}  "
              f"95%CI {_ci_str(row['ece_hb_nstar_ci_lo'], row['ece_hb_nstar_ci_hi'])}  "
              f"Imp {row['improvement_hb_mean']:+.4f} {_ci_str(row['improvement_hb_ci_lo'], row['improvement_hb_ci_hi'])}")
        print(f"  IR  @ N*={n_star:<4d}  ECE: {row['ece_ir_nstar_mean']:.4f}  "
              f"95%CI {_ci_str(row['ece_ir_nstar_ci_lo'], row['ece_ir_nstar_ci_hi'])}  "
              f"Imp {row['improvement_ir_mean']:+.4f} {_ci_str(row['improvement_ir_ci_lo'], row['improvement_ir_ci_hi'])}")

    if not all_rows:
        print("\nNo results collected. Check --results_dir.")
        return

    print_summary(all_rows)
    save_outputs(all_rows, args.output_dir)
    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
