"""
experiment_precision_ablation.py

Discretization precision ablation for the calibration map.

For each bin width s in BIN_WIDTHS, re-runs the full calibration pipeline
(joint distribution estimation → minimax solve → apply map → evaluate)
on the three model pairs (Qwen3-8B, LLaMA-3.1-8B, Ministral-3-8B)
for MMLU-Redux, MMLU, and CommonSenseQA.

Also evaluates fixed (pre-computed) methods — Base, Instruct, TS, HB, IR,
MaxConf — which are not re-run per bin_width (their outputs are independent
of discretization precision).  These are stored once per (dataset, model_pair)
with bin_width=null.  Including them here allows side-by-side comparison of
Ours at every bin_width against fixed baselines.

Records both calibration performance (ECE, Brier, accuracy) AND joint
distribution statistics at two levels:

  Full-report statistics (from the (j,m) report space used by the algorithm):
    n_r1, n_r2   — unique (j,m) reports for instruct and base
    nnz          — non-zero cells in P(r1,r2)
    total_cells  — n_r1 × n_r2 (full grid size)
    max_report_frac — fraction at most common (j,m) report
    n_conf_by_class — per-class unique confidence bins (JSON only)

  Confidence-marginal statistics (collapsing over class j):
    n_unique_conf_r1, n_unique_conf_r2 — unique confidence values m
    Note: n_r1 ≤ K × n_unique_conf_r1 because each confidence bin can
    appear under multiple class labels.

Usage:
    python source/summarize+figures/experiment_precision_ablation.py \\
        --results_dir ./results \\
        --output_dir  ./results/precision_ablation
"""

from src.experiments.cli import add_selection_arguments, selected_pairs

from src.experiments.io import load_pair_data, load_jsonl
from src.experiments.settings import SHORT_NAME as MODEL_SHORTNAME
from src.config import MAIN_PAIRS
from src.config import DATASET_CONFIG
from src.experiments.pipeline import apply_calibration_map
from src.config import MODEL_PAIRS
from src.metrics import ece_uwb
MODELS_ORDER = [inst for inst, _ in MODEL_PAIRS]

import os
import json
import argparse
import csv
import time as _time
from collections import Counter

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from source.post_processing.utils import calculate_joint_distribution_for_confidence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIN_WIDTHS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

# Model pairs per dataset

# Calibration split → application split, template names per dataset.
# Per paper Appendix F.1:
#   MMLU-Redux: calibrate on MMLU/validation (n=1531), evaluate on MMLU-Redux/test (n=5700)
#   MMLU:       calibrate on MMLU/validation (n=1531), evaluate on MMLU/test (n=14,042)
#   CommonSenseQA: calibrate on CSQA/train (n=9741), evaluate on CSQA/validation (n=1221)

# Short display name for plots/tables

DATASET_SHORTNAME = {
    "MMLU_Redux":    "MMLU-Redux",
    "MMLU":          "MMLU",
    "CommonSenseQA": "CommonSenseQA",
}

# Ordered list of datasets for plots — one column each
DATASETS_PLOT_ORDER = ["MMLU_Redux", "MMLU", "CommonSenseQA"]


# ---------------------------------------------------------------------------
# Fixed (pre-computed) method helpers
# ---------------------------------------------------------------------------

# Methods whose outputs are pre-computed and do NOT change with bin_width.
# Their rows appear once per (dataset, model_pair) with bin_width=null.
FIXED_METHODS = ["Base", "Instruct", "TS", "HB", "IR", "MaxConf"]

METHOD_CATEGORY = {
    "Base":     "label-free",
    "Instruct": "label-free",
    "Ours":     "label-free",
    "TS":       "supervised",
    "HB":       "supervised",
    "IR":       "supervised",
    "MaxConf":  "aggregation",
}


def _fixed_method_paths(results_dir, dataset, instruct_slug, base_slug):
    """
    Build file paths for all fixed methods on the apply split.
    Paths may not exist — caller must check.
    """
    cfg          = DATASET_CONFIG[dataset]
    apply_dataset = cfg.get("apply_dataset", dataset)
    apply_split  = cfg["apply_split"]
    it           = cfg["instruct_tpl"]
    bt           = cfg["base_tpl"]
    calib        = f"calibrated_on_{cfg['calib_dataset']}_{cfg['calib_split']}"
    pp_file      = f"{it}_{base_slug}_{bt}.jsonl"

    def _pp(method_dir):
        return os.path.join(results_dir, apply_dataset, apply_split,
                            "post_processing", method_dir, "confidence",
                            "logprob", instruct_slug, pp_file)

    return {
        "Base":     os.path.join(results_dir, apply_dataset, apply_split,
                                 "LLMs", "confidence", "logprob",
                                 base_slug, f"{bt}.jsonl"),
        "Instruct": os.path.join(results_dir, apply_dataset, apply_split,
                                 "LLMs", "confidence", "logprob",
                                 instruct_slug, f"{it}.jsonl"),
        "TS":       _pp("temperature_scaling"),
        "HB":       _pp("histogram_binning"),
        "IR":       _pp("isotonic_regression"),
        "MaxConf":  _pp("max_confidence"),
    }


def evaluate_fixed_method(results_dir, dataset, instruct_slug, base_slug, method_name):
    """
    Load a pre-computed method's .jsonl on the apply split and compute metrics.
    Returns a dict {method, category, accuracy, ece_uwb_10, brier_score, n_test}
    or None if the file is missing / empty.
    """
    paths = _fixed_method_paths(results_dir, dataset, instruct_slug, base_slug)
    path  = paths.get(method_name)
    if path is None or not os.path.exists(path):
        return None
    records = load_jsonl(path)
    if not records:
        return None
    c, z = _extract_cz(records)
    se_acc  = se_accuracy(z)
    se_ece  = se_ece_uwb(c, z, n_bins=10)
    se_bs   = se_brier_score(c, z)
    return {
        "method":           method_name,
        "category":         METHOD_CATEGORY.get(method_name, "other"),
        "accuracy":         float(z.mean()),
        "se_acc":           se_acc,
        "ece_uwb_10":       ece_uwb(records, n_bins=10),
        "se_ece_uwb_10":    se_ece,
        "brier_score":      brier_score(records),
        "se_brier_score":   se_bs,
        "n_test":           len(records),
    }


def load_calib_and_apply_data(results_dir, dataset, instruct_slug, base_slug):
    """
    Load the four data lists needed for calibration + application.
    Returns (calib_inst, calib_base, apply_inst, apply_base) or None on failure.

    Per paper Appendix F.1, calibration and application may come from different datasets:
      MMLU-Redux: calibrate on MMLU/validation, apply to MMLU-Redux/test
      CommonSenseQA: calibrate on CSQA/train, apply to CSQA/validation
    """
    cfg = DATASET_CONFIG[dataset]
    calib_dataset = cfg["calib_dataset"]
    apply_dataset = cfg.get("apply_dataset", dataset)
    calib_inst, calib_base, apply_inst, apply_base = load_pair_data(
        results_dir, dataset, instruct_slug, base_slug)
    if not calib_inst or not calib_base or not apply_inst or not apply_base:
        missing = []
        if not calib_inst:  missing.append(f"calib_inst ({calib_dataset}/{cfg['calib_split']})")
        if not calib_base:  missing.append(f"calib_base ({calib_dataset}/{cfg['calib_split']})")
        if not apply_inst:  missing.append(f"apply_inst ({apply_dataset}/{cfg['apply_split']})")
        if not apply_base:  missing.append(f"apply_base ({apply_dataset}/{cfg['apply_split']})")
        print(f"    [SKIP] Missing data: {', '.join(missing)}")
        return None
    return calib_inst, calib_base, apply_inst, apply_base


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def brier_score(records, K=4):
    """Mean Brier Score for K-class multi-choice setting.
    Each record has 'choices', 'gold', and 'model_prediction' with 'final_answer'+'confidence'.
    We use the one-vs-rest Brier Score: (conf_correct - 1)^2 + sum_{wrong} conf_wrong^2.
    Since we only have the top-predicted confidence, approximate with
    BS ≈ (conf - is_correct)^2 where conf is the reported confidence.
    """
    bs_list = []
    for r in records:
        gold      = r.get("gold", -1)
        choices   = r.get("choices", [])
        pred      = r.get("model_prediction", {})
        ans       = pred.get("final_answer", "")
        conf      = float(pred.get("confidence", 0.0))
        is_correct = int(len(choices) > 0 and choices[gold] == ans)
        bs_list.append((conf - is_correct) ** 2)
    return float(np.mean(bs_list)) if bs_list else float("nan")


def accuracy(records):
    if not records:
        return float("nan")
    return float(np.mean([
        r["choices"][r["gold"]] == r["model_prediction"]["final_answer"]
        for r in records
    ]))


# ---------------------------------------------------------------------------
# Standard error helpers  (match calculate_standard_errors.py exactly)
# ---------------------------------------------------------------------------

def _extract_cz(records):
    """Return (confidence c, correctness z) as numpy arrays."""
    c = np.array([float(r["model_prediction"]["confidence"]) for r in records])
    z = np.array([
        float(r["choices"][r["gold"]] == r["model_prediction"]["final_answer"])
        for r in records
    ])
    return c, z


def se_accuracy(z):
    """SE(ACC) = sqrt(acc * (1 - acc) / n)  — binomial formula."""
    acc = float(z.mean())
    return math.sqrt(acc * (1.0 - acc) / len(z))


def se_brier_score(c, z):
    """SE(BS) = std((c_i - z_i)^2, ddof=1) / sqrt(n)."""
    b = (c - z) ** 2
    return float(np.std(b, ddof=1) / math.sqrt(len(b)))


def se_ece_uwb(c, z, n_bins=10):
    """Delta-method SE for UWB ECE.

    SE = (1/n) * sqrt(sum_m( n_m * Var(d_i | bin m, ddof=1) ))
    where d_i = z_i - c_i.  Bins with < 2 samples contribute 0.
    """
    d = z - c
    n = len(c)
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(bin_limits[:-1], bin_limits[1:])):
        mask = (c <= hi) if i == 0 else ((c > lo) & (c <= hi))
        n_m = int(mask.sum())
        if n_m < 2:
            continue
        total += n_m * float(np.var(d[mask], ddof=1))
    return float(math.sqrt(total) / n)


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run_one(dataset, instruct_slug, base_slug, bin_width, results_dir, diagnostics_path=None):
    """
    Runs the calibration pipeline at a single bin_width for one model pair.

    Returns a dict with:
      - grid_stats from calculate_joint_distribution_for_confidence
      - solver_stats from solve_minimax_sparse_vectorized (via transformation)
      - performance metrics on the apply split
    or None on failure.
    """
    data = load_calib_and_apply_data(results_dir, dataset, instruct_slug, base_slug)
    if data is None:
        return None
    calib_inst, calib_base, apply_inst, apply_base = data

    # --- Build joint distribution on calibration split ---
    # The joint distribution is over report pairs r=(j,m) where j=predicted class
    # and m=discretized confidence.  grid_stats reflects this full (j,m) space.
    jd_calib = calculate_joint_distribution_for_confidence(
        calib_inst, calib_base, bin_width=bin_width
    )
    if jd_calib is None:
        print("    [SKIP] Failed to build calibration joint distribution.")
        return None

    # --- Build application-split joint distribution (for apply_inst lookup) ---
    jd_apply = calculate_joint_distribution_for_confidence(
        apply_inst, apply_base, bin_width=bin_width
    )
    if jd_apply is None:
        print("    [SKIP] Failed to build application joint distribution.")
        return None

    # --- Always-computed stats (grid, extra, missing rate) ---
    grid_stats = jd_calib.get("grid_stats", {})
    rp         = jd_calib.get("report_pairs", [])   # list of ((j1,m1), (j2,m2))
    K          = jd_calib.get("K", 4)
    r1_j_vals  = jd_calib["r1_j"]
    r1_m_vals  = jd_calib["r1_m"]

    # max_report_frac: fraction at the most common instruct (j,m) report.
    if rp:
        r1_counts = Counter(p[0] for p in rp)
        max_report_frac = max(r1_counts.values()) / len(rp)
    else:
        max_report_frac = float("nan")

    # n_unique_conf_r1 / n_unique_conf_r2: unique confidence values m,
    # marginalizing over class j.  Contrast with n_r1/n_r2 (unique (j,m) reports).
    unique_r1 = set(p[0] for p in rp)
    unique_r2 = set(p[1] for p in rp)
    n_unique_conf_r1 = len(set(m for _, m in unique_r1))
    n_unique_conf_r2 = len(set(m for _, m in unique_r2))

    # Per-class unique confidence bins (JSON only).
    n_conf_by_class = {
        "inst": {str(j): len(set(m for jj, m in unique_r1 if jj == j)) for j in range(K)},
        "base": {str(j): len(set(m for jj, m in unique_r2 if jj == j)) for j in range(K)},
    }

    # support_sample_hist: {sample_count: num_cells}
    # For each non-zero cell (r1, r2) in the joint distribution, count how many
    # calibration samples contributed to it.  The histogram answers:
    # "given a cell with k samples, how many such cells exist?"
    # This quantifies how well the calibration set covers the grid.
    #   - Many cells with count=1 → sparse, high variance estimates
    #   - Cells with count >> 1 → well-estimated mass concentrations
    pair_counts = Counter(rp)                          # (r1, r2) → sample count
    cell_count_hist = Counter(pair_counts.values())    # sample_count → num_cells
    support_sample_hist = {
        str(k): v for k, v in sorted(cell_count_hist.items())
    }

    # Support miss: test reports outside exact calibration support. Retain the
    # original statistic and sample set. Normally deployed misses keep their
    # quantized report; miss rates from other wrappers' filtering/skipped sets
    # must not be equated with a fallback rate among their final outputs.
    calib_report_set = set(zip(r1_j_vals.tolist(), r1_m_vals.tolist()))
    test_r1_reports  = [p[0] for p in jd_apply["report_pairs"]]
    n_test_total     = len(test_r1_reports)
    n_miss           = sum(1 for r in test_r1_reports if r not in calib_report_set)
    missing_rate     = n_miss / n_test_total if n_test_total > 0 else float("nan")

    extra = {
        "max_report_frac":      max_report_frac,
        "n_unique_conf_r1":     n_unique_conf_r1,
        "n_unique_conf_r2":     n_unique_conf_r2,
        "n_conf_by_class":      n_conf_by_class,       # nested dict; JSON only
        "support_sample_hist":  support_sample_hist,   # nested dict; JSON only
    }

    # --- Solve and apply ---
    diagnostic_args = {"diagnostics_path": diagnostics_path} if diagnostics_path is not None else {}
    updated_records, solver_stats = apply_calibration_map(
        jd_calib, jd_apply, precision=4, **diagnostic_args
    )

    if updated_records is None or not updated_records:
        # Solver failed (LP infeasible or numerical failure) — store row with NaN
        # metrics so the failure is visible in the output rather than silently dropped.
        print(f"    [SOLVER FAILED] status={solver_stats.get('solver_status', '?') if solver_stats else '?'}"
              f"  n_r1={grid_stats.get('n_r1','?')}  n_r2={grid_stats.get('n_r2','?')}")
        metrics = {
            "accuracy":         float("nan"),
            "se_acc":           float("nan"),
            "ece_uwb_10":       float("nan"),
            "se_ece_uwb_10":    float("nan"),
            "brier_score":      float("nan"),
            "se_brier_score":   float("nan"),
            "n_test":           0,
            "n_test_total":     n_test_total,
            "n_miss":           n_miss,
            "missing_rate":     missing_rate,
        }
    else:
        # --- Evaluate ---
        c, z = _extract_cz(updated_records)
        metrics = {
            "accuracy":         float(z.mean()),
            "se_acc":           se_accuracy(z),
            "ece_uwb_10":       ece_uwb(updated_records, n_bins=10),
            "se_ece_uwb_10":    se_ece_uwb(c, z, n_bins=10),
            "brier_score":      brier_score(updated_records),
            "se_brier_score":   se_brier_score(c, z),
            "n_test":           len(updated_records),
            "n_test_total":     n_test_total,
            "n_miss":           n_miss,
            "missing_rate":     missing_rate,
        }

    return {
        "method":   "Ours",
        "category": "label-free",
        **grid_stats,
        **(solver_stats or {}),
        **metrics,
        **extra,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
LINESTYLES = ["-", "--", ":", "-."]


def _rows_for(all_rows, dataset, instruct, method="Ours"):
    """Filter rows for a specific dataset/model/method combination.
    Plots of bin_width-varying results should always pass method='Ours'.
    """
    return [r for r in all_rows
            if r["dataset"] == dataset and r["instruct"] == instruct
            and r.get("method") == method]


def plot_performance(all_rows, output_dir):
    """ECE and Brier Score vs bin_width, one subplot per dataset."""
    n_ds = len(DATASETS_PLOT_ORDER)
    fig, axes = plt.subplots(2, n_ds, figsize=(6 * n_ds, 8))

    for col, dataset in enumerate(DATASETS_PLOT_ORDER):
        ds_name = DATASET_SHORTNAME[dataset]
        for metric, ax, ylabel in [
            ("ece_uwb_10", axes[0, col], "ECE (UWB-10)"),
            ("brier_score", axes[1, col], "Brier Score"),
        ]:
            ax.set_title(f"{ds_name} — {ylabel}")
            ax.set_xlabel("Bin width s")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.invert_xaxis()

            for instruct, color, ls in zip(MODELS_ORDER, COLORS, LINESTYLES):
                rows = _rows_for(all_rows, dataset, instruct)
                if not rows:
                    continue
                rows = sorted(rows, key=lambda r: r["bin_width"])
                xs = [r["bin_width"] for r in rows if r.get(metric) is not None]
                ys = [r[metric] for r in rows if r.get(metric) is not None]
                label = MODEL_SHORTNAME.get(instruct, instruct)
                ax.plot(xs, ys, marker="o", color=color, ls=ls, label=label)

            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "performance_vs_s.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_grid_stats(all_rows, output_dir):
    """nnz, n_vars, solver_time vs bin_width (log-log), one panel per dataset."""
    n_ds = len(DATASETS_PLOT_ORDER)
    fig, axes = plt.subplots(3, n_ds, figsize=(6 * n_ds, 12))

    stat_configs = [
        ("nnz",           "Non-zero cells (NNZ)"),
        ("n_vars",        "CVXPY variables (K × NNZ)"),
        ("solver_time_s", "Solver time (s)"),
    ]

    for col, dataset in enumerate(DATASETS_PLOT_ORDER):
        ds_name = DATASET_SHORTNAME[dataset]
        for row_idx, (stat, ylabel) in enumerate(stat_configs):
            ax = axes[row_idx, col]
            ax.set_title(f"{ds_name} — {ylabel}")
            ax.set_xlabel("Bin width s")
            ax.set_ylabel(ylabel)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.invert_xaxis()

            for instruct, color, ls in zip(MODELS_ORDER, COLORS, LINESTYLES):
                rows = _rows_for(all_rows, dataset, instruct)
                if not rows:
                    continue
                rows = sorted(rows, key=lambda r: r["bin_width"])
                xs = [r["bin_width"] for r in rows if r.get(stat) is not None]
                ys = [r[stat] for r in rows if r.get(stat) is not None]
                if not xs:
                    continue
                label = MODEL_SHORTNAME.get(instruct, instruct)
                ax.plot(xs, ys, marker="o", color=color, ls=ls, label=label)

            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(output_dir, "grid_stats_vs_s.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_sparsity(all_rows, output_dir):
    """Sparsity (NNZ / total_cells) and fill_rate vs bin_width."""
    n_ds = len(DATASETS_PLOT_ORDER)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 4))

    for col, dataset in enumerate(DATASETS_PLOT_ORDER):
        ax = axes[col]
        ds_name = DATASET_SHORTNAME[dataset]
        ax.set_title(f"{ds_name} — Grid sparsity")
        ax.set_xlabel("Bin width s")
        ax.set_ylabel("NNZ / total_cells")
        ax.set_xscale("log")
        ax.invert_xaxis()

        for instruct, color, ls in zip(MODELS_ORDER, COLORS, LINESTYLES):
            rows = _rows_for(all_rows, dataset, instruct)
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r["bin_width"])
            xs = [r["bin_width"] for r in rows if r.get("sparsity") is not None]
            ys = [r["sparsity"] for r in rows if r.get("sparsity") is not None]
            if not xs:
                continue
            label = MODEL_SHORTNAME.get(instruct, instruct)
            ax.plot(xs, ys, marker="o", color=color, ls=ls, label=label)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "sparsity_vs_s.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_missing_rate(all_rows, output_dir):
    """
    Missing rate (fraction of test records outside calibration support) vs bin_width.
    Only Ours rows have this statistic.
    """
    n_ds = len(DATASETS_PLOT_ORDER)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 4))

    for col, dataset in enumerate(DATASETS_PLOT_ORDER):
        ax = axes[col]
        ds_name = DATASET_SHORTNAME[dataset]
        ax.set_title(f"{ds_name} — Missing rate vs s")
        ax.set_xlabel("Bin width s")
        ax.set_ylabel("Fraction of test records outside calibration support")
        ax.set_xscale("log")
        ax.invert_xaxis()

        for instruct, color, ls in zip(MODELS_ORDER, COLORS, LINESTYLES):
            rows = _rows_for(all_rows, dataset, instruct)
            if not rows:
                continue
            rows = sorted(rows, key=lambda r: r["bin_width"])
            xs = [r["bin_width"] for r in rows if r.get("missing_rate") is not None
                  and not (isinstance(r["missing_rate"], float) and r["missing_rate"] != r["missing_rate"])]
            ys = [r["missing_rate"] for r in rows if r.get("missing_rate") is not None
                  and not (isinstance(r["missing_rate"], float) and r["missing_rate"] != r["missing_rate"])]
            if not xs:
                continue
            label = MODEL_SHORTNAME.get(instruct, instruct)
            ax.plot(xs, ys, marker="o", color=color, ls=ls, label=label)

        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    path = os.path.join(output_dir, "missing_rate_vs_s.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_scatter_vs_default(all_rows, output_dir, default_s=0.01):
    """
    For each non-default bin_width, scatter ECE@s vs ECE@default_s.
    Points near the diagonal = stable under bin_width change.
    """
    other_widths = [s for s in BIN_WIDTHS if s != default_s]
    if not other_widths:
        return

    fig, axes = plt.subplots(1, len(other_widths), figsize=(4 * len(other_widths), 4))
    if len(other_widths) == 1:
        axes = [axes]

    for ax, s_other in zip(axes, other_widths):
        xs, ys, colors_pts = [], [], []
        for dataset in DATASETS_PLOT_ORDER:
            for instruct, color in zip(MODELS_ORDER, COLORS):
                rows_def   = [r for r in _rows_for(all_rows, dataset, instruct)
                               if abs(r["bin_width"] - default_s) < 1e-9]
                rows_other = [r for r in _rows_for(all_rows, dataset, instruct)
                               if abs(r["bin_width"] - s_other) < 1e-9]
                if rows_def and rows_other:
                    xs.append(rows_def[0].get("ece_uwb_10", float("nan")))
                    ys.append(rows_other[0].get("ece_uwb_10", float("nan")))
                    colors_pts.append(color)

        ax.scatter(xs, ys, c=colors_pts, s=60, zorder=3)
        lim_max = max(max(xs, default=0.3), max(ys, default=0.3)) * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, label="y=x")
        ax.set_xlabel(f"ECE at s={default_s}")
        ax.set_ylabel(f"ECE at s={s_other}")
        ax.set_title(f"s={s_other} vs s={default_s}")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "scatter_s001_vs_others.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "dataset", "instruct", "base",
    # Method identifier
    "method", "category",
    # bin_width is set for Ours; null/empty for fixed methods
    "bin_width",
    # Full-report grid stats (n_r1/n_r2 = unique (j,m) reports, not confidence bins)
    # These are only populated for method=Ours
    "n_calib", "n_r1", "n_r2", "total_cells", "nnz", "sparsity", "fill_rate",
    "n_vars", "solver_time_s", "solver_status",
    # Confidence-marginal stats (Ours only)
    "n_unique_conf_r1", "n_unique_conf_r2",
    # Concentration diagnostic (full-report level, Ours only)
    "max_report_frac",
    # Performance — all methods (point estimates + delta-method SEs)
    "n_test", "accuracy", "se_acc", "ece_uwb_10", "se_ece_uwb_10",
    "brier_score", "se_brier_score",
    # Missing-rate statistics (method=Ours only)
    # n_test_total = raw test records before any skips
    # n_miss       = test records whose (j,m) is outside the calibration support
    # missing_rate = n_miss / n_test_total
    "n_test_total", "n_miss", "missing_rate",
    # n_conf_by_class and support_sample_hist are nested dicts — stored in JSON only
]


def save_outputs(all_rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # JSON — includes all fields, including n_conf_by_class nested dict
    json_path = os.path.join(output_dir, "summary_table.json")
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"Saved: {json_path}")

    # CSV
    csv_path = os.path.join(output_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Discretization precision ablation experiment.")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir",  default="./results/precision_ablation")
    parser.add_argument("--bin_widths", nargs="+", type=float, default=BIN_WIDTHS,
                        help="List of bin widths to evaluate.")
    add_selection_arguments(parser)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    n_pairs = len(selected_pairs(MAIN_PAIRS, args))
    n_bw    = len(args.bin_widths)
    # Total work = one fixed-methods pass per pair + one Ours run per (pair, bin_width)
    total = n_pairs * (len(FIXED_METHODS) + n_bw)
    done  = 0

    for dataset, instruct_slug, base_slug in selected_pairs(MAIN_PAIRS, args):
        ds_short    = DATASET_SHORTNAME[dataset]
        model_short = MODEL_SHORTNAME.get(instruct_slug, instruct_slug)

        # ------------------------------------------------------------------
        # Fixed methods (pre-computed, independent of bin_width)
        # ------------------------------------------------------------------
        print(f"\n{'='*70}")
        print(f"{ds_short} | {model_short} | Fixed methods (Base, Instruct, TS, HB, IR, MaxConf)")
        print(f"{'='*70}")

        for method_name in FIXED_METHODS:
            done += 1
            result = evaluate_fixed_method(
                args.results_dir, dataset, instruct_slug, base_slug, method_name
            )
            if result is None:
                print(f"  [{done}/{total}] {method_name}: SKIP (file not found or empty)")
                continue
            row = {
                "dataset":   dataset,
                "instruct":  instruct_slug,
                "base":      base_slug,
                "bin_width": None,   # not applicable for fixed methods
                **result,
            }
            all_rows.append(row)
            print(
                f"  [{done}/{total}] {method_name:8s}  "
                f"acc={row['accuracy']:.4f}  "
                f"ECE={row['ece_uwb_10']:.4f}  "
                f"Brier={row['brier_score']:.4f}  "
                f"n={row['n_test']}"
            )

        # ------------------------------------------------------------------
        # Ours — re-run at each bin_width
        # ------------------------------------------------------------------
        for bin_width in args.bin_widths:
            done += 1
            label = (f"[{done}/{total}] {ds_short} | {model_short} | Ours | s={bin_width}")
            print(f"\n{'-'*70}")
            print(label)

            t0 = _time.time()
            diagnostics_path = os.path.join(
                args.output_dir, "diagnostics", dataset,
                f"{instruct_slug}__{base_slug}__s-{bin_width}.json",
            )
            stats = run_one(dataset, instruct_slug, base_slug, bin_width, args.results_dir,
                            diagnostics_path=diagnostics_path)
            elapsed = _time.time() - t0

            if stats is None:
                print(f"  FAILED (wall time {elapsed:.1f}s)")
                continue

            row = {
                "dataset":   dataset,
                "instruct":  instruct_slug,
                "base":      base_slug,
                "bin_width": bin_width,
                **stats,
            }
            all_rows.append(row)
            def _fmt(v, fmt=".4f"):
                return ("nan" if (isinstance(v, float) and v != v) else format(v, fmt))
            mr  = row.get('missing_rate', float('nan'))
            ece = row.get('ece_uwb_10',  float('nan'))
            bs  = row.get('brier_score', float('nan'))
            print(
                f"  OK  n_r1={row.get('n_r1','?')}  n_r2={row.get('n_r2','?')}  "
                f"nnz={row.get('nnz','?')}  n_vars={row.get('n_vars','?')}  "
                f"sparsity={row.get('sparsity', 0):.4f}  "
                f"miss={_fmt(mr)} ({row.get('n_miss','?')}/{row.get('n_test_total','?')})  "
                f"solve={row.get('solver_time_s', 0):.1f}s  "
                f"status={row.get('solver_status','?')}  "
                f"ECE={_fmt(ece)}  Brier={_fmt(bs)}  wall={elapsed:.1f}s"
            )

    if not all_rows:
        print("\nNo successful results. Check paths and data.")
        return

    save_outputs(all_rows, args.output_dir)
    plot_performance(all_rows, args.output_dir)
    plot_grid_stats(all_rows, args.output_dir)
    plot_sparsity(all_rows, args.output_dir)
    plot_missing_rate(all_rows, args.output_dir)
    plot_scatter_vs_default(all_rows, args.output_dir, default_s=0.01)

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
