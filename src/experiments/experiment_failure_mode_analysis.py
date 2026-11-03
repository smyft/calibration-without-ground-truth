"""
experiment_failure_mode_analysis.py

Failure-mode analysis of the reference and calibration report distributions.

For every (Instruct, Base) model pair on MMLU-Redux and MMLU (val→test), computes:
  - Standard ECE / accuracy metrics for Base, Instruct, Ours, TS, HB, IR
  - Structural diagnostics from the calibration set (MMLU/validation) at two levels:

    (a) Confidence-marginal diagnostics (operate on confidence scores m alone):
        max_conf_bin_frac     — fraction of calibration samples at the most common
                                discretized confidence value m (s=0.01)
        n_unique_conf_inst    — number of unique confidence bins for instruct
        n_unique_conf_base    — number of unique confidence bins for base
        tv_conf               — TV distance between instruct and base confidence
                                histograms (1D marginal on [0,1])
        wass_dist             — Wasserstein-1 distance between 1D confidence CDFs
        base_mean_conf_calib  — mean base confidence on the calibration set
        frac_inst_above_09    — fraction of instruct confidences > 0.9

    (b) Full-report diagnostics (operate on report pairs r=(j,m) where
        j=predicted class index, m=discretized confidence):
        max_report_frac       — fraction of calibration samples at the most common
                                instruct (j,m) report; the correct report-level
                                concentration metric used in the calibration algorithm
        n_unique_inst_reports — number of unique instruct (j,m) reports = n_r1 in
                                the joint distribution grid
        n_unique_base_reports — number of unique base (j,m) reports = n_r2
        tv_report             — TV distance between empirical marginal distributions
                                over instruct reports P(r1) and base reports P(r2)
        jd_nnz                — non-zero cells in P(r1,r2) joint distribution matrix
        n_reports_by_class    — per-class count of unique confidence bins for each
                                model (dict: {"inst": {j: count}, "base": {j: count}})

    The calibration algorithm works on the full (j,m) report space, so the
    full-report diagnostics are the primary indicators of calibration difficulty.
    The confidence-marginal diagnostics are kept for interpretability and
    1D visualization.

  - Output-quality metrics on the test set:
      discrimination, mode_frac, conf_gap

Produces correlation scatter plots, recalibration map visualizations,
reliability diagrams, and a summary CSV + JSON.

Usage:
    python source/summarize+figures/experiment_failure_mode_analysis.py \\
        --results_dir ./results \\
        --output_dir  ./results/failure_mode_analysis
"""

from src.experiments.cli import add_selection_arguments, selected_models

from src.experiments.io import load_jsonl
from src.config import SHORT_NAME

import os
import json
import argparse
import csv
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wasserstein_distance

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

from source.post_processing.utils import calculate_joint_distribution_for_confidence


# ---------------------------------------------------------------------------
# Dataset and model configuration
# ---------------------------------------------------------------------------

# Datasets to analyse
DATASETS = ["MMLU_Redux", "MMLU"]

DATASET_CONFIG = {
    "MMLU_Redux": {
        "test_dataset":  "MMLU_Redux",
        "test_split":    "test",
        "calib_dataset": "MMLU",
        "calib_split":   "validation",
        "calib_tag":     "calibrated_on_MMLU_validation",
        "label":         "MMLU-Redux",
    },
    "MMLU": {
        "test_dataset":  "MMLU",
        "test_split":    "test",
        "calib_dataset": "MMLU",
        "calib_split":   "validation",
        "calib_tag":     "calibrated_on_MMLU_validation",
        "label":         "MMLU",
    },
}

# All pairs with available data

INSTRUCT_TPL = "baseline_thinking"
BASE_TPL     = "baseline"

# Highlight focal pairs for detailed plots
FOCAL_FAILURE = "meta-llama_Llama-3.1-8B-Instruct"
FOCAL_SUCCESS = "Qwen_Qwen3-8B"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _conf_arr(records):
    return np.array([float(r["model_prediction"]["confidence"]) for r in records])


def _correct_arr(records):
    return np.array([
        float(r["choices"][r["gold"]] == r["model_prediction"]["final_answer"])
        for r in records
    ])


def load_test_files(results_dir, dataset, inst_slug, base_slug):
    """Load (inst, base, ours, ts, hb, ir) confidence files from the test split."""
    cfg       = DATASET_CONFIG[dataset]
    td        = os.path.join(results_dir, cfg["test_dataset"], cfg["test_split"])
    calib_tag = cfg["calib_tag"]
    pp_file   = f"{INSTRUCT_TPL}_{base_slug}_{BASE_TPL}.jsonl"

    def _pp(method_dir):
        return os.path.join(td, "post_processing", method_dir,
                            "confidence", "logprob", inst_slug, pp_file)

    inst = load_jsonl(os.path.join(td, "LLMs", "confidence", "logprob",
                                   inst_slug, f"{INSTRUCT_TPL}.jsonl"))
    base = load_jsonl(os.path.join(td, "LLMs", "confidence", "logprob",
                                   base_slug, f"{BASE_TPL}.jsonl"))
    ours = load_jsonl(os.path.join(td, "post_processing", "transformation",
                                   "confidence", calib_tag, "confidence",
                                   "logprob", inst_slug, pp_file))
    ts   = load_jsonl(_pp("temperature_scaling"))
    hb   = load_jsonl(_pp("histogram_binning"))
    ir   = load_jsonl(_pp("isotonic_regression"))
    return inst, base, ours, ts, hb, ir


def load_calib_files(results_dir, dataset, inst_slug, base_slug):
    """Load (inst, base) confidence files from the calibration split."""
    cfg = DATASET_CONFIG[dataset]
    cd  = os.path.join(results_dir, cfg["calib_dataset"], cfg["calib_split"],
                       "LLMs", "confidence", "logprob")
    inst = load_jsonl(os.path.join(cd, inst_slug, f"{INSTRUCT_TPL}.jsonl"))
    base = load_jsonl(os.path.join(cd, base_slug, f"{BASE_TPL}.jsonl"))
    return inst, base


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

ECE_BINS = 10

# Sweep ranges for sensitivity analysis
ECE_BIN_SWEEP = [5, 10, 15, 20, 25]           # B values for ECE binning sensitivity
JD_STEP_SWEEP = [0.1, 0.05, 0.02, 0.01, 0.005]  # s values for joint-distribution discretization sensitivity


def ece_uwb_with_bins(is_correct, confidences, n_bins=ECE_BINS):
    """
    Compute UWB ECE and return per-bin statistics.

    Returns
    -------
    ece_value : float
    bins : list[dict]
        One dict per non-empty bin:
        {bin_idx, lo, hi, n, mean_conf, mean_acc, abs_gap}
        n is the sample count — the primary quantity the user wants to track.
    """
    is_correct  = np.asarray(is_correct,  dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    if len(confidences) == 0:
        return float("nan"), []
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_out = []
    for i, (lo, hi) in enumerate(zip(bin_limits[:-1], bin_limits[1:])):
        mask = (confidences <= hi) if i == 0 else ((confidences > lo) & (confidences <= hi))
        n = int(mask.sum())
        if n == 0:
            continue
        mc = float(confidences[mask].mean())
        ma = float(is_correct[mask].mean())
        ag = float(abs(mc - ma))
        ece += ag * n / len(confidences)
        bins_out.append({
            "bin_idx":   i,
            "lo":        round(float(lo), 6),
            "hi":        round(float(hi), 6),
            "n":         n,
            "mean_conf": round(mc, 6),
            "mean_acc":  round(ma, 6),
            "abs_gap":   round(ag, 6),
        })
    return float(ece), bins_out


def ece_uwb(is_correct, confidences, n_bins=ECE_BINS):
    is_correct  = np.asarray(is_correct,  dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    if len(confidences) == 0:
        return float("nan")
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bin_limits[:-1], bin_limits[1:])):
        mask = (confidences <= hi) if i == 0 else ((confidences > lo) & (confidences <= hi))
        if mask.sum() == 0:
            continue
        ece += abs(confidences[mask].mean() - is_correct[mask].mean()) * mask.sum() / len(confidences)
    return float(ece)


def total_variation(confs1, confs2, n_bins=20):
    """TV distance between two confidence arrays, histogrammed into n_bins bins on [0,1]."""
    bins = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    h1, _ = np.histogram(confs1, bins=bins, density=False)
    h2, _ = np.histogram(confs2, bins=bins, density=False)
    h1 = h1 / h1.sum() if h1.sum() > 0 else h1
    h2 = h2 / h2.sum() if h2.sum() > 0 else h2
    return float(0.5 * np.sum(np.abs(h1 - h2)))


def max_conf_bin_fraction(confs, s=0.01):
    """Fraction of samples at the most common discretized confidence value m.
    Operates on the confidence marginal only — ignores predicted class j."""
    discretized = np.round(np.round(confs / s) * s, 10)
    counts = Counter(discretized)
    return max(counts.values()) / len(discretized)


def max_report_fraction(report_pairs_r1, total):
    """Fraction of calibration samples at the most common instruct report (j, m).
    report_pairs_r1: list of (j, m) tuples — the r1 side of each calibration pair.
    This is the correct report-level concentration metric that matches the
    joint distribution P(r1, r2) used by the calibration algorithm."""
    counts = Counter(report_pairs_r1)
    return max(counts.values()) / total if total > 0 else float("nan")


def discrimination(is_correct, confidences):
    """Mean confidence of correct predictions minus mean of incorrect predictions."""
    is_correct  = np.asarray(is_correct, dtype=bool)
    confidences = np.asarray(confidences, dtype=float)
    if is_correct.sum() == 0 or (~is_correct).sum() == 0:
        return float("nan")
    return float(confidences[is_correct].mean() - confidences[~is_correct].mean())


def mode_fraction(confidences):
    """Fraction of samples at the single most common (rounded to 4dp) confidence."""
    rounded = np.round(confidences, 4)
    counts  = Counter(rounded)
    return max(counts.values()) / len(rounded)


def per_bin_calibration(is_correct, confidences, bin_width=0.05):
    """Return list of (lo, mean_conf, mean_acc, n) for each non-empty bin."""
    edges = np.arange(0.0, 1.0 + bin_width, bin_width)
    out = []
    for lo in edges[:-1]:
        hi = lo + bin_width
        mask = (confidences >= lo) & (confidences < hi + 1e-9)
        if mask.sum() == 0:
            continue
        out.append((lo, float(confidences[mask].mean()),
                    float(is_correct[mask].mean()), int(mask.sum())))
    return out


# ---------------------------------------------------------------------------
# Per-pair analysis
# ---------------------------------------------------------------------------

def analyze_pair(results_dir, dataset, inst_slug, base_slug):
    """Return a results dict for one (dataset, inst, base) pair, or None on skip."""
    # Load test files
    inst_r, base_r, ours_r, ts_r, hb_r, ir_r = load_test_files(
        results_dir, dataset, inst_slug, base_slug)
    if not inst_r or not base_r:
        print(f"  [SKIP] Missing test data for {inst_slug}")
        return None

    # Load calibration files
    calib_inst_r, calib_base_r = load_calib_files(results_dir, dataset, inst_slug, base_slug)
    have_calib = bool(calib_inst_r and calib_base_r)

    # ---- Test-set arrays ----
    inst_conf    = _conf_arr(inst_r)
    inst_correct = _correct_arr(inst_r)
    base_conf    = _conf_arr(base_r)
    base_correct = _correct_arr(base_r)

    ours_conf    = _conf_arr(ours_r)    if ours_r else np.array([])
    ours_correct = _correct_arr(ours_r) if ours_r else np.array([])
    ts_conf      = _conf_arr(ts_r)      if ts_r   else np.array([])
    ts_correct   = _correct_arr(ts_r)   if ts_r   else np.array([])
    hb_conf      = _conf_arr(hb_r)      if hb_r   else np.array([])
    hb_correct   = _correct_arr(hb_r)   if hb_r   else np.array([])
    ir_conf      = _conf_arr(ir_r)      if ir_r   else np.array([])
    ir_correct   = _correct_arr(ir_r)   if ir_r   else np.array([])

    # ---- ECE metrics ----
    # NOTE: our recalibration method outputs a joint (class, confidence) pair —
    # it can change the predicted answer, not just the confidence.  Therefore
    # ece_ours must use ours_correct (correctness of the recalibrated prediction),
    # not inst_correct.  TS preserves the argmax class, so ts_correct == inst_correct
    # up to dropped predictions, but we use ts_correct for consistency.
    ece_inst = ece_uwb(inst_correct, inst_conf)
    ece_base = ece_uwb(base_correct, base_conf)
    ece_ours = ece_uwb(ours_correct, ours_conf) if len(ours_conf) else float("nan")
    ece_ts   = ece_uwb(ts_correct,   ts_conf)   if len(ts_conf)   else float("nan")
    ece_hb   = ece_uwb(hb_correct,   hb_conf)   if len(hb_conf)   else float("nan")
    ece_ir   = ece_uwb(ir_correct,   ir_conf)   if len(ir_conf)   else float("nan")
    delta    = float(ece_ts - ece_ours) if not (np.isnan(ece_ts) or np.isnan(ece_ours)) else float("nan")
    delta_hb = float(ece_hb - ece_ours) if not (np.isnan(ece_hb) or np.isnan(ece_ours)) else float("nan")
    delta_ir = float(ece_ir - ece_ours) if not (np.isnan(ece_ir) or np.isnan(ece_ours)) else float("nan")

    # ---- ECE sweep: bin counts at each (method, B) combination ----
    # For each of four methods and each B in ECE_BIN_SWEEP, store ECE value
    # and per-bin statistics (n, mean_conf, mean_acc, abs_gap).  This lets
    # us see how samples distribute across confidence bins for each method
    # and how that distribution changes with bin width — key to understanding
    # failure modes such as mass concentration in a single bin.
    _sweep_inputs = {
        "Instruct": (inst_correct, inst_conf),
        "Base":     (base_correct, base_conf),
        "Ours":     (ours_correct, ours_conf) if len(ours_conf) else (np.array([]), np.array([])),
        "TS":       (ts_correct,   ts_conf)   if len(ts_conf)   else (np.array([]), np.array([])),
        "HB":       (hb_correct,   hb_conf)   if len(hb_conf)   else (np.array([]), np.array([])),
        "IR":       (ir_correct,   ir_conf)   if len(ir_conf)   else (np.array([]), np.array([])),
    }
    ece_sweep = {}
    for mname, (corr_arr, conf_arr) in _sweep_inputs.items():
        ece_sweep[mname] = {}
        for B in ECE_BIN_SWEEP:
            ece_val, bins = ece_uwb_with_bins(corr_arr, conf_arr, n_bins=B)
            ece_sweep[mname][B] = {"ece": ece_val, "bins": bins}

    # ---- JD sweep: joint distribution stats at each discretization step s ----
    # Initialised here; filled inside the `if have_calib:` block below.
    jd_sweep = {}

    acc_inst = float(inst_correct.mean())
    acc_base = float(base_correct.mean())

    # ---- Calibration-set structural diagnostics ----
    # (a) Confidence-marginal diagnostics (on m only)
    max_conf_bin_frac = float("nan")
    base_mean_conf_calib = float("nan")
    calib_acc_inst = float("nan")
    tv_conf = float("nan")
    wass_dist = float("nan")
    n_unique_conf_inst = float("nan")
    n_unique_conf_base = float("nan")
    frac_inst_above_09_calib = float("nan")
    # (b) Full-report diagnostics (on (j,m) pairs)
    max_report_frac = float("nan")
    n_unique_inst_reports = float("nan")
    n_unique_base_reports = float("nan")
    tv_report = float("nan")
    jd_nnz = float("nan")
    n_reports_by_class = None   # stored in JSON only

    if have_calib:
        calib_ic = _conf_arr(calib_inst_r)
        calib_bc = _conf_arr(calib_base_r)
        calib_ic_corr = _correct_arr(calib_inst_r)

        # --- (a) confidence-marginal metrics ---
        max_conf_bin_frac        = max_conf_bin_fraction(calib_ic, s=0.01)
        base_mean_conf_calib     = float(calib_bc.mean())
        calib_acc_inst           = float(calib_ic_corr.mean())
        tv_conf                  = total_variation(calib_ic, calib_bc, n_bins=20)
        wass_dist                = float(wasserstein_distance(calib_ic, calib_bc))
        frac_inst_above_09_calib = float(np.mean(calib_ic > 0.9))
        disc_calib_i             = np.round(np.round(calib_ic / 0.01) * 0.01, 10)
        n_unique_conf_inst       = len(set(disc_calib_i))
        disc_calib_b             = np.round(np.round(calib_bc / 0.01) * 0.01, 10)
        n_unique_conf_base       = len(set(disc_calib_b))

        # --- (b) full-report metrics via calculate_joint_distribution_for_confidence ---
        # The calibration algorithm operates on report pairs r=(j,m), so these
        # metrics are the primary indicators of calibration difficulty.
        jd = calculate_joint_distribution_for_confidence(
            calib_inst_r, calib_base_r, precision=2
        )
        if jd is not None:
            gs  = jd["grid_stats"]
            rp  = jd["report_pairs"]   # list of ((j1,m1), (j2,m2))
            K   = jd["K"]

            n_unique_inst_reports = gs["n_r1"]
            n_unique_base_reports = gs["n_r2"]
            jd_nnz                = gs["nnz"]

            # max_report_frac: most common instruct (j,m) report
            max_report_frac = max_report_fraction([p[0] for p in rp], len(rp))

            # TV over full (j,m) report distributions P(r1) vs P(r2)
            r1_counts = Counter(p[0] for p in rp)
            r2_counts = Counter(p[1] for p in rp)
            all_reports = set(r1_counts) | set(r2_counts)
            n_total = len(rp)
            tv_report = 0.5 * sum(
                abs(r1_counts.get(r, 0) / n_total - r2_counts.get(r, 0) / n_total)
                for r in all_reports
            )

            # Per-class unique confidence bins (for JSON storage)
            r1_unique = set(p[0] for p in rp)
            r2_unique = set(p[1] for p in rp)
            n_reports_by_class = {
                "inst": {str(j): len([r for r in r1_unique if r[0] == j])
                         for j in range(K)},
                "base": {str(j): len([r for r in r2_unique if r[0] == j])
                         for j in range(K)},
            }

        # ---- JD sweep: recompute joint distribution at each step size s ----
        # Records how the support size (n_r1, n_r2), fill (nnz, sparsity), and
        # mass concentration (max_report_frac) change with the discretization
        # step.  Coarser s → fewer unique reports → denser joint distribution
        # → potentially different LP behaviour and ECE.
        for s in JD_STEP_SWEEP:
            jd_s = calculate_joint_distribution_for_confidence(
                calib_inst_r, calib_base_r, bin_width=s
            )
            if jd_s is None:
                jd_sweep[str(s)] = None
                continue
            gs_s  = jd_s["grid_stats"]
            rp_s  = jd_s["report_pairs"]
            max_rf_s = (max_report_fraction([p[0] for p in rp_s], len(rp_s))
                        if rp_s else float("nan"))
            jd_sweep[str(s)] = {
                "n_r1":            gs_s.get("n_r1"),
                "n_r2":            gs_s.get("n_r2"),
                "total_cells":     gs_s.get("total_cells"),
                "nnz":             gs_s.get("nnz"),
                "sparsity":        gs_s.get("sparsity"),
                "max_report_frac": float(max_rf_s),
            }

    # ---- Test-set output analysis ----
    # Discrimination: mean_conf(correct) − mean_conf(incorrect).
    # For Ours, correctness is determined by the recalibrated predicted answer
    # (which may differ from the instruct's answer).
    disc_ours     = discrimination(ours_correct, ours_conf) if len(ours_conf) else float("nan")
    disc_ts       = discrimination(ts_correct,   ts_conf)   if len(ts_conf)   else float("nan")
    disc_hb       = discrimination(hb_correct,   hb_conf)   if len(hb_conf)   else float("nan")
    disc_ir       = discrimination(ir_correct,   ir_conf)   if len(ir_conf)   else float("nan")
    disc_inst     = discrimination(inst_correct, inst_conf)
    disc_base     = discrimination(base_correct, base_conf)

    mode_frac_ours = mode_fraction(ours_conf) if len(ours_conf) else float("nan")
    mode_frac_ts   = mode_fraction(ts_conf)   if len(ts_conf)   else float("nan")
    mode_frac_hb   = mode_fraction(hb_conf)   if len(hb_conf)   else float("nan")
    mode_frac_ir   = mode_fraction(ir_conf)   if len(ir_conf)   else float("nan")

    n_unique_ours  = len(set(np.round(ours_conf, 4))) if len(ours_conf) else 0
    n_unique_ts    = len(set(np.round(ts_conf, 4)))   if len(ts_conf)   else 0
    n_unique_hb    = len(set(np.round(hb_conf,  4)))  if len(hb_conf)   else 0
    n_unique_ir    = len(set(np.round(ir_conf,  4)))  if len(ir_conf)   else 0

    # conf_gap = mean_conf − own_accuracy.  Ours may have a different accuracy
    # than Instruct because the recalibrated map can change the predicted answer.
    acc_ours = float(ours_correct.mean()) if len(ours_correct) else float("nan")
    acc_ts   = float(ts_correct.mean())   if len(ts_correct)   else float("nan")
    acc_hb   = float(hb_correct.mean())   if len(hb_correct)   else float("nan")
    acc_ir   = float(ir_correct.mean())   if len(ir_correct)   else float("nan")
    conf_gap_ours  = float(ours_conf.mean() - acc_ours) if len(ours_conf) else float("nan")
    conf_gap_ts    = float(ts_conf.mean()   - acc_ts)   if len(ts_conf)   else float("nan")
    conf_gap_hb    = float(hb_conf.mean()   - acc_hb)   if len(hb_conf)   else float("nan")
    conf_gap_ir    = float(ir_conf.mean()   - acc_ir)   if len(ir_conf)   else float("nan")

    # ---- Per-bin calibration data (for plotting) ----
    per_bin_ours = per_bin_calibration(ours_correct, ours_conf) if len(ours_conf) else []
    per_bin_ts   = per_bin_calibration(ts_correct,   ts_conf)   if len(ts_conf)   else []

    return {
        "dataset":    dataset,
        "dataset_label": DATASET_CONFIG[dataset]["label"],
        "inst_slug": inst_slug,
        "base_slug": base_slug,
        "short_name": SHORT_NAME.get(inst_slug, inst_slug),
        # Basic metrics
        "acc_inst": acc_inst,
        "acc_base": acc_base,
        "acc_gap": float(acc_inst - acc_base),
        # acc_ours/hb/ir may differ from acc_inst because our recalibration map can
        # change the predicted answer (argmax of gamma_a in confidence.py).
        "acc_ours": acc_ours,
        "acc_ts":   acc_ts,
        "acc_hb":   acc_hb,
        "acc_ir":   acc_ir,
        "ece_inst": ece_inst,
        "ece_base": ece_base,
        "ece_ours": ece_ours,
        "ece_ts":   ece_ts,
        "ece_hb":   ece_hb,
        "ece_ir":   ece_ir,
        "delta":    delta,     # ece_ts - ece_ours: positive = ours better
        "delta_hb": delta_hb,  # ece_hb - ece_ours: positive = ours better
        "delta_ir": delta_ir,  # ece_ir - ece_ours: positive = ours better
        "ours_wins": (delta > 0) if not np.isnan(delta) else None,
        # ---------------------------------------------------------------
        # Calibration-set structural diagnostics
        # (a) Confidence-marginal diagnostics (operate on confidence m only)
        # ---------------------------------------------------------------
        "max_conf_bin_frac": max_conf_bin_frac,       # renamed from max_inst_bin_frac
        "base_mean_conf_calib": base_mean_conf_calib,
        "calib_acc_inst": calib_acc_inst,
        "base_calib_gap": float(base_mean_conf_calib - calib_acc_inst)
                          if not np.isnan(base_mean_conf_calib) else float("nan"),
        "tv_conf": tv_conf,                           # renamed from tv_distance
        "wass_dist": wass_dist,
        "frac_inst_above_09_calib": frac_inst_above_09_calib,
        "n_unique_conf_inst": n_unique_conf_inst,     # renamed from n_unique_inst_bins
        "n_unique_conf_base": n_unique_conf_base,     # renamed from n_unique_base_bins
        # ---------------------------------------------------------------
        # (b) Full-report diagnostics (operate on report pairs r=(j,m))
        # These match the joint distribution P(r1,r2) used by the algorithm
        # ---------------------------------------------------------------
        "max_report_frac": max_report_frac,           # primary concentration metric
        "n_unique_inst_reports": n_unique_inst_reports,  # = n_r1 in grid
        "n_unique_base_reports": n_unique_base_reports,  # = n_r2 in grid
        "tv_report": tv_report,                       # TV over full (j,m) distributions
        "jd_nnz": jd_nnz,                             # nnz in P(r1,r2) matrix
        "n_reports_by_class": n_reports_by_class,     # {inst/base: {class: count}}
        # ---------------------------------------------------------------
        # Test-set output analysis
        # ---------------------------------------------------------------
        "disc_inst": disc_inst,
        "disc_base": disc_base,
        "disc_ours": disc_ours,
        "disc_ts":   disc_ts,
        "disc_hb":   disc_hb,
        "disc_ir":   disc_ir,
        "mode_frac_ours": mode_frac_ours,
        "mode_frac_ts":   mode_frac_ts,
        "mode_frac_hb":   mode_frac_hb,
        "mode_frac_ir":   mode_frac_ir,
        "n_unique_ours": n_unique_ours,
        "n_unique_ts":   n_unique_ts,
        "n_unique_hb":   n_unique_hb,
        "n_unique_ir":   n_unique_ir,
        "conf_gap_ours": conf_gap_ours,
        "conf_gap_ts":   conf_gap_ts,
        "conf_gap_hb":   conf_gap_hb,
        "conf_gap_ir":   conf_gap_ir,
        # ---------------------------------------------------------------
        # Sensitivity sweeps (JSON-only; not written to CSV)
        # ---------------------------------------------------------------
        # ece_sweep[method][B] = {"ece": float, "bins": [{bin_idx, lo, hi, n, mean_conf, mean_acc, abs_gap}]}
        # jd_sweep[str(s)]     = {"n_r1", "n_r2", "total_cells", "nnz", "sparsity", "max_report_frac"}
        "ece_sweep": ece_sweep,
        "jd_sweep":  jd_sweep,
        # Raw arrays for plotting (not written to CSV)
        "_inst_conf": inst_conf,
        "_inst_correct": inst_correct,
        "_base_conf": base_conf,
        "_base_correct": base_correct,
        "_ours_conf": ours_conf,
        "_ours_correct": ours_correct,
        "_ts_conf": ts_conf,
        "_ts_correct": ts_correct,
        "_calib_ic": _conf_arr(calib_inst_r) if have_calib else np.array([]),
        "_calib_bc": _conf_arr(calib_base_r) if have_calib else np.array([]),
        "_hb_conf":    hb_conf,
        "_hb_correct": hb_correct,
        "_ir_conf":    ir_conf,
        "_ir_correct": ir_correct,
        "_per_bin_ours": per_bin_ours,
        "_per_bin_ts":   per_bin_ts,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {
    "success": "#2ca02c",
    "failure": "#d62728",
    "tie":     "#ff7f0e",
}


def _color(row):
    if row["ours_wins"] is True:  return COLORS["success"]
    if row["ours_wins"] is False: return COLORS["failure"]
    return COLORS["tie"]


def _label(row):
    return row["short_name"]


def scatter_with_spearman(ax, rows, x_key, y_key="delta",
                          xlabel=None, ylabel="ECE_TS − ECE_Ours (↑ = ours better)"):
    """Generic scatter: x=x_key, y=delta, colored by success/failure."""
    xs, ys, labels, colors = [], [], [], []
    for r in rows:
        xv = r.get(x_key, float("nan"))
        yv = r.get(y_key, float("nan"))
        if np.isnan(xv) or np.isnan(yv):
            continue
        xs.append(xv); ys.append(yv)
        labels.append(_label(r)); colors.append(_color(r))

    ax.scatter(xs, ys, c=colors, s=90, zorder=3, edgecolors="k", linewidths=0.4)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, ls="--", lw=1, color="gray", alpha=0.6)

    if len(xs) >= 3:
        rho, pval = spearmanr(xs, ys)
        ax.set_title(f"ρ={rho:+.2f}, p={pval:.3f}", fontsize=9)

    ax.set_xlabel(xlabel or x_key, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_scatter_grid(rows, output_dir):
    """Scatter panels: delta vs each diagnostic, covering both marginal and full-report metrics."""
    predictors = [
        # Confidence-marginal diagnostics
        ("base_mean_conf_calib",  "Base mean confidence (calib, conf-marginal)"),
        ("max_conf_bin_frac",     "Max confidence bin fraction (calib, conf-marginal)"),
        ("tv_conf",               "TV: Inst vs Base confidence histograms (marginal)"),
        ("wass_dist",             "Wasserstein-1: Inst vs Base confidences (marginal)"),
        # Full-report diagnostics
        ("max_report_frac",       "Max report (j,m) fraction (calib, full-report)"),
        ("tv_report",             "TV: Inst vs Base report dists P(r1) vs P(r2)"),
        ("n_unique_inst_reports", "Unique instruct (j,m) reports n_r1 (calib)"),
        # ECE / accuracy
        ("ece_inst",              "ECE of Instruct model (test set)"),
        ("acc_gap",               "Accuracy gap: Acc_Inst − Acc_Base"),
        # Output quality
        ("disc_ours",             "Discrimination score of Ours"),
        ("conf_gap_ours",         "Confidence gap: mean_conf(Ours) − acc_ours"),
    ]
    n = len(predictors)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, predictors):
        scatter_with_spearman(ax, rows, key, xlabel=label)

    for ax in axes[n:]:
        ax.set_visible(False)

    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=COLORS["success"], label="Ours wins (ECE_ours < ECE_ts)"),
        mpatches.Patch(color=COLORS["failure"], label="Ours fails (ECE_ours > ECE_ts)"),
        mpatches.Patch(color=COLORS["tie"],     label="≈ Tie"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(output_dir, "scatter_all_predictors.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_reliability_diagram(ax, is_correct, confidences, title, n_bins=10, color="steelblue"):
    """Plot calibration curve and confidence histogram on ax."""
    is_correct  = np.asarray(is_correct, dtype=float)
    confidences = np.asarray(confidences, dtype=float)

    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    mid_confs, mid_accs, fracs = [], [], []
    for i, (lo, hi) in enumerate(zip(bin_limits[:-1], bin_limits[1:])):
        mask = (confidences <= hi) if i == 0 else ((confidences > lo) & (confidences <= hi))
        if mask.sum() == 0:
            continue
        mid_confs.append(confidences[mask].mean())
        mid_accs.append(is_correct[mask].mean())
        fracs.append(mask.sum() / len(confidences))

    # Calibration curve
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect")
    ax.bar(mid_confs, mid_accs, width=1 / n_bins * 0.8, alpha=0.7,
           color=color, label="accuracy")

    # Histogram inset
    ax2 = ax.twinx()
    ax2.hist(confidences, bins=n_bins, range=(0, 1), alpha=0.25,
             color="gray", density=True)
    ax2.set_ylabel("density", fontsize=7, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
    ax2.set_ylim(0, ax2.get_ylim()[1] * 4)  # reduce visual weight of histogram

    ece = ece_uwb(is_correct, confidences, n_bins=n_bins)
    acc = is_correct.mean()
    ax.set_title(f"{title}\nECE={ece:.3f}, Acc={acc:.3f}", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.grid(True, alpha=0.2)


def plot_reliability_4panel(row, output_dir, tag):
    """4-panel reliability diagram for one model: Base / Instruct / Ours / TS."""
    if len(row["_ours_conf"]) == 0:
        print(f"  [SKIP] No Ours data for reliability diagram ({tag})")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    sname = row["short_name"]

    # Each method uses its own correctness array.  Ours can have a different
    # predicted answer than Instruct (the recalibration map outputs argmax(gamma_a)
    # which may differ from the original predicted class), so _ours_correct ≠
    # _inst_correct in general.  TS preserves the argmax, so _ts_correct ==
    # _inst_correct[:len(_ts_correct)], but we use _ts_correct for consistency.
    plot_reliability_diagram(axes[0], row["_base_correct"],  row["_base_conf"],  "Base",     color="#aec6cf")
    plot_reliability_diagram(axes[1], row["_inst_correct"],  row["_inst_conf"],  "Instruct", color="#f4a460")
    plot_reliability_diagram(axes[2], row["_ours_correct"],  row["_ours_conf"],  "Ours",     color="#2ca02c")
    if len(row["_ts_conf"]):
        plot_reliability_diagram(axes[3], row["_ts_correct"], row["_ts_conf"],   "TS",       color="#9467bd")
    else:
        axes[3].text(0.5, 0.5, "No TS data", ha="center", va="center", fontsize=12)
        axes[3].set_visible(True)

    fig.suptitle(f"Reliability Diagrams — {sname}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(output_dir, f"reliability_{tag}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_recalibration_map(row, output_dir, tag, s=0.01):
    """
    Two-panel figure for one model:
    Left:  scatter of (inst_conf_discretized → output_conf), bubble size ∝ sample fraction
    Right: 2D histogram of (inst_conf, base_conf) on calibration set (joint distribution)
    """
    calib_ic = row.get("_calib_ic", np.array([]))
    calib_bc = row.get("_calib_bc", np.array([]))
    inst_conf = row["_inst_conf"]
    ours_conf = row["_ours_conf"]

    if len(ours_conf) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sname = row["short_name"]

    # --- Left: recalibration map ---
    ax = axes[0]
    disc_inst = np.round(np.round(inst_conf / s) * s, 10)
    mapping_pairs = list(zip(disc_inst[:len(ours_conf)], ours_conf))
    from collections import defaultdict
    bucket = defaultdict(list)
    for di, oc in mapping_pairs:
        bucket[di].append(oc)

    xs, ys, sizes = [], [], []
    for di, oc_list in sorted(bucket.items()):
        xs.append(di)
        ys.append(np.mean(oc_list))
        sizes.append(len(oc_list))

    total = len(ours_conf)
    sizes_norm = [400 * sz / total for sz in sizes]
    sc = ax.scatter(xs, ys, s=sizes_norm, alpha=0.7, color="#2ca02c", edgecolors="k", lw=0.5, zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="y=x")
    ax.axhline(row["acc_inst"], ls=":", lw=1.5, color="red", alpha=0.7,
               label=f"Test acc = {row['acc_inst']:.3f}")
    ax.set_xlabel("Instruct confidence (discretized, s=0.01)", fontsize=9)
    ax.set_ylabel("Output confidence (mean of bucket)", fontsize=9)
    ax.set_title(f"{sname}: Recalibration Map\n"
                 f"(bubble size ∝ sample fraction)", fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotate max-frac bucket
    if sizes:
        max_i = np.argmax(sizes)
        ax.annotate(f"{sizes[max_i]/total:.1%}", (xs[max_i], ys[max_i]),
                    fontsize=8, color="darkgreen",
                    xytext=(5, -12), textcoords="offset points")

    # --- Right: confidence-marginal heatmap P(m1, m2) on calibration set ---
    # NOTE: This heatmap shows the confidence-marginal P(m1, m2) — the 2D histogram
    # of (instruct_conf, base_conf) summed over all class pairs (j1, j2).
    # The actual joint distribution used by the calibration algorithm is P(r1, r2)
    # over report pairs r=(j,m), which has n_r1 × n_r2 cells vs the K² × (1/s)²
    # full grid. The confidence marginal shown here provides a visual summary of
    # where the two models' confidences concentrate.
    ax2 = axes[1]
    if len(calib_ic) > 0 and len(calib_bc) > 0:
        bins = np.linspace(0, 1, 21)  # 20 bins = 0.05 width
        H, xedges, yedges = np.histogram2d(calib_ic, calib_bc, bins=bins)
        H = H / H.sum()
        im = ax2.imshow(H.T, origin="lower", aspect="auto", cmap="Blues",
                        extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax2, label="P(m1, m2)")
        ax2.set_xlabel("Instruct confidence m (calibration set)", fontsize=9)
        ax2.set_ylabel("Base confidence m (calibration set)", fontsize=9)
        mcbf  = row.get("max_conf_bin_frac",  float("nan"))
        mrf   = row.get("max_report_frac",    float("nan"))
        n_r1  = row.get("n_unique_inst_reports", float("nan"))
        ax2.set_title(
            f"{sname}: Confidence marginal P(m1,m2)\n"
            f"base_mean={row['base_mean_conf_calib']:.3f}  "
            f"max_conf_bin={mcbf:.3f}  max_report={mrf:.3f}  n_r1={int(n_r1) if not np.isnan(n_r1) else 'N/A'}",
            fontsize=8)
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.plot([0, 1], [0, 1], "r--", lw=1, alpha=0.5)
        ax2.grid(True, alpha=0.2)
    else:
        ax2.text(0.5, 0.5, "No calibration data", ha="center", va="center")
        ax2.set_title(f"{sname}: Confidence marginal P(m1,m2) (no calib data)")

    fig.tight_layout()
    path = os.path.join(output_dir, f"map_{tag}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_confidence_histograms(rows, output_dir):
    """Stacked panels: instruct vs base confidence histograms for all pairs."""
    valid = [r for r in rows if len(r["_inst_conf"]) > 0 and len(r["_base_conf"]) > 0]
    n = len(valid)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()

    bins = np.linspace(0, 1, 21)
    for ax, r in zip(axes, valid):
        ax.hist(r["_inst_conf"], bins=bins, alpha=0.55, color="#f4a460",
                density=True, label="Instruct")
        ax.hist(r["_base_conf"], bins=bins, alpha=0.55, color="#aec6cf",
                density=True, label="Base")
        sname = r["short_name"]
        tv = r.get("tv_distance", float("nan"))
        wins = r.get("ours_wins")
        status = "✓" if wins else ("✗" if wins is False else "~")
        ax.set_title(f"{sname} {status}\nTV={tv:.3f}", fontsize=8)
        ax.set_xlabel("Confidence", fontsize=7)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

    axes[0].legend(fontsize=7)
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "confidence_histograms.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_bin_calibration(rows, output_dir):
    """Per-output-bin calibration (acc vs conf) for failure and success case."""
    focal = [r for r in rows
             if r["inst_slug"] in (FOCAL_FAILURE, FOCAL_SUCCESS)
             and len(r.get("_per_bin_ours", [])) > 0]

    if not focal:
        return

    fig, axes = plt.subplots(1, len(focal), figsize=(7 * len(focal), 5))
    if len(focal) == 1:
        axes = [axes]

    for ax, r in zip(axes, focal):
        pb = r["_per_bin_ours"]
        xs  = [b[1] for b in pb]  # mean_conf
        ys  = [b[2] for b in pb]  # mean_acc
        ns  = [b[3] for b in pb]  # count
        total = sum(ns)
        sizes = [300 * n / total for n in ns]

        ax.scatter(xs, ys, s=sizes, alpha=0.7, color="#2ca02c", edgecolors="k", lw=0.5, zorder=3)
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"n={n}", (x, y), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfect calibration")
        ax.axhline(r["acc_inst"], ls=":", lw=1.5, color="red",
                   label=f"Test accuracy={r['acc_inst']:.3f}")
        ax.set_xlabel("Mean calibrated confidence (within bin)", fontsize=9)
        ax.set_ylabel("Fraction correct", fontsize=9)
        ax.set_title(f"{r['short_name']} — Per-bin calibration (Ours)\n"
                     f"ECE_ours={r['ece_ours']:.3f}, ECE_ts={r['ece_ts']:.3f}", fontsize=9)
        ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "per_bin_calibration.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_compact_reliability_grid(rows, output_dir, dataset="MMLU_Redux"):
    """Compact grid: one row per model, columns = Base/Instruct/Ours/TS/HB/IR."""
    valid = [r for r in rows
             if r.get("dataset") == dataset and len(r["_ours_conf"]) > 0]
    n = len(valid)
    if n == 0:
        return

    method_names = ["Base", "Instruct", "Ours", "TS", "HB", "IR"]
    colors       = ["#aec6cf", "#f4a460", "#2ca02c", "#9467bd", "#17becf", "#bcbd22"]

    fig, axes = plt.subplots(n, len(method_names), figsize=(4 * len(method_names), 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row_i, r in enumerate(valid):
        # Each method uses its own correctness array. Ours/HB/IR can have different
        # predicted answers from Instruct; TS preserves the argmax.
        method_data = [
            (r["_base_correct"],  r["_base_conf"]),
            (r["_inst_correct"],  r["_inst_conf"]),
            (r["_ours_correct"],  r["_ours_conf"]),
            (r["_ts_correct"],    r["_ts_conf"]),
            (r["_hb_correct"],    r["_hb_conf"]),
            (r["_ir_correct"],    r["_ir_conf"]),
        ]
        status = "✓ wins" if r["ours_wins"] else ("✗ fails" if r["ours_wins"] is False else "≈ tie")
        for col_i, ((corr, conf), mname, color) in enumerate(zip(method_data, method_names, colors)):
            ax = axes[row_i, col_i]
            if len(conf) == 0:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                continue
            plot_reliability_diagram(ax, corr, conf, mname, n_bins=10, color=color)
            if col_i == 0:
                ax.set_ylabel(f"{r['short_name']}\n({status})\n\nFraction Correct", fontsize=8)

    ds_label = DATASET_CONFIG[dataset]["label"]
    fig.suptitle(f"Reliability Diagrams — All Model Pairs ({ds_label})", fontsize=13, y=1.01)
    fig.tight_layout()
    safe_ds = dataset.lower().replace("_", "")
    path = os.path.join(output_dir, f"reliability_all_{safe_ds}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Sensitivity plot helpers
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "Instruct": "#f4a460",
    "Base":     "#aec6cf",
    "Ours":     "#2ca02c",
    "TS":       "#9467bd",
    "HB":       "#17becf",
    "IR":       "#bcbd22",
}


def _focal_rows(all_rows, dataset="MMLU_Redux"):
    """Return (failure_row, success_row) for the two focal models on the given dataset."""
    pool = [r for r in all_rows if r.get("dataset") == dataset]
    fail = next((r for r in pool if r["inst_slug"] == FOCAL_FAILURE), None)
    succ = next((r for r in pool if r["inst_slug"] == FOCAL_SUCCESS), None)
    return [r for r in [fail, succ] if r is not None]


def plot_ece_bin_counts_focal(all_rows, output_dir):
    """
    For the two focal models (failure + success), show how many test samples
    fall into each ECE bin for each of the four methods (at default B=ECE_BINS).

    Bar height = sample count (n).  Bars are annotated with the bin's |conf−acc|
    gap so the reader can see simultaneously where mass concentrates and where
    it is miscalibrated.
    """
    focal = _focal_rows(all_rows)
    if not focal:
        return

    methods = ["Instruct", "Base", "Ours", "TS", "HB", "IR"]
    B = ECE_BINS
    nrows, ncols = len(focal), len(methods)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)

    for row_i, r in enumerate(focal):
        sname  = r["short_name"]
        status = "✓ wins" if r["ours_wins"] else ("✗ fails" if r["ours_wins"] is False else "≈ tie")
        for col_i, mname in enumerate(methods):
            ax   = axes[row_i][col_i]
            data = r.get("ece_sweep", {}).get(mname, {}).get(B)
            if data is None or not data.get("bins"):
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_title(f"{mname} — N/A", fontsize=9)
                continue

            bins    = data["bins"]
            ece_val = data["ece"]
            xs   = [(b["lo"] + b["hi"]) / 2 for b in bins]
            ns   = [b["n"]        for b in bins]
            gaps = [b["abs_gap"]  for b in bins]
            width = 1.0 / B * 0.85

            bars = ax.bar(xs, ns, width=width, color=METHOD_COLORS.get(mname, "gray"),
                          alpha=0.75, edgecolor="k", linewidth=0.4)
            # Annotate each bar: n on top, gap below
            for bar, n_val, gap_val in zip(bars, ns, gaps):
                bx = bar.get_x() + bar.get_width() / 2
                bh = bar.get_height()
                ax.text(bx, bh + max(ns) * 0.01, f"n={n_val}", ha="center",
                        va="bottom", fontsize=5.5, rotation=90)
                ax.text(bx, bh / 2, f"{gap_val:.3f}", ha="center",
                        va="center", fontsize=5.5, color="white",
                        fontweight="bold")

            ax.set_xlabel("Confidence bin center", fontsize=8)
            ax.set_ylabel("# samples", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_title(f"{mname}  ECE={ece_val:.4f}\n{sname} ({status})", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"ECE bin sample counts (B={B}, test set) — focal models", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "ece_bin_counts_focal.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_ece_sensitivity_focal(all_rows, output_dir):
    """
    For the two focal models, plot ECE (UWB) vs number of bins B for all four
    methods.  Reveals whether the failure mode persists across binning choices.
    """
    focal = _focal_rows(all_rows)
    if not focal:
        return

    methods = ["Instruct", "Base", "Ours", "TS", "HB", "IR"]
    markers = {"Instruct": "o", "Base": "s", "Ours": "^", "TS": "D", "HB": "P", "IR": "X"}
    ncols = len(focal)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5), squeeze=False)

    for col_i, r in enumerate(focal):
        ax     = axes[0][col_i]
        sname  = r["short_name"]
        status = "✓ wins" if r["ours_wins"] else ("✗ fails" if r["ours_wins"] is False else "≈ tie")
        for mname in methods:
            sweep = r.get("ece_sweep", {}).get(mname, {})
            xs = sorted(b for b in sweep if isinstance(b, int))
            ys = [sweep[b]["ece"] for b in xs]
            valid = [(x, y) for x, y in zip(xs, ys) if not np.isnan(y)]
            if not valid:
                continue
            xv, yv = zip(*valid)
            ax.plot(xv, yv, marker=markers.get(mname, "o"), color=METHOD_COLORS.get(mname, "gray"),
                    linewidth=1.5, markersize=5, label=mname)

        ax.set_xlabel("Number of ECE bins B", fontsize=9)
        ax.set_ylabel("ECE (UWB)", fontsize=9)
        ax.set_title(f"{sname} ({status})\nECE vs ECE bin count B", fontsize=9)
        ax.set_xticks(ECE_BIN_SWEEP)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("ECE sensitivity to binning choice — focal models (test set)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "ece_sensitivity_focal.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_jd_sensitivity_focal(all_rows, output_dir):
    """
    For the two focal models, plot how the joint distribution statistics change
    with the discretization step size s.

    Two metrics shown:
      - n_r1: number of unique instruct (j,m) reports (= LP variable count)
      - max_report_frac: fraction of calibration samples in the densest cell
      - nnz: non-zero entries in P(r1,r2)

    Reveals whether the failure mode is robust to the discretization choice.
    """
    focal = _focal_rows(all_rows)
    if not focal:
        return

    metrics = [
        ("n_r1",            "Unique instruct reports n_r1",   False),
        ("max_report_frac", "Max report fraction",            False),
        ("nnz",             "Non-zero cells in P(r1,r2)",     False),
        ("sparsity",        "Sparsity of P(r1,r2)",           False),
    ]
    nrows, ncols = len(metrics), len(focal)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for col_i, r in enumerate(focal):
        sname  = r["short_name"]
        status = "✓ wins" if r["ours_wins"] else ("✗ fails" if r["ours_wins"] is False else "≈ tie")
        jd_sw  = r.get("jd_sweep", {})
        s_vals = sorted(float(k) for k in jd_sw if jd_sw[k] is not None)

        for row_i, (key, ylabel, logy) in enumerate(metrics):
            ax  = axes[row_i][col_i]
            ys  = [jd_sw[str(s)][key] for s in s_vals if jd_sw.get(str(s)) and jd_sw[str(s)].get(key) is not None]
            xs  = [s for s in s_vals if jd_sw.get(str(s)) and jd_sw[str(s)].get(key) is not None]
            if not xs:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue
            ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=1.5, markersize=5)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.3g}", (x, y), fontsize=7,
                            xytext=(0, 5), textcoords="offset points", ha="center")
            # Mark the default s=0.01 used in paper experiments
            ax.axvline(0.01, ls="--", lw=1, color="red", alpha=0.5, label="default s=0.01")
            ax.set_xlabel("Discretization step s", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{sname} ({status})\n{ylabel} vs s", fontsize=9)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.invert_xaxis()   # coarse → fine left→right reads naturally
            ax.grid(True, alpha=0.3, which="both")
            if row_i == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Joint distribution sensitivity to discretization step s — focal models",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(output_dir, "jd_sensitivity_focal.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "dataset", "dataset_label", "short_name", "inst_slug", "base_slug",
    "acc_inst", "acc_base", "acc_gap",
    "acc_ours", "acc_ts", "acc_hb", "acc_ir",
    "ece_inst", "ece_base", "ece_ours", "ece_ts", "ece_hb", "ece_ir",
    "delta", "delta_hb", "delta_ir", "ours_wins",
    # Confidence-marginal diagnostics
    "max_conf_bin_frac", "base_mean_conf_calib", "calib_acc_inst",
    "base_calib_gap", "tv_conf", "wass_dist",
    "frac_inst_above_09_calib", "n_unique_conf_inst", "n_unique_conf_base",
    # Full-report diagnostics
    "max_report_frac", "n_unique_inst_reports", "n_unique_base_reports",
    "tv_report", "jd_nnz",
    # Test-set output analysis
    "disc_inst", "disc_base", "disc_ours", "disc_ts", "disc_hb", "disc_ir",
    "mode_frac_ours", "mode_frac_ts", "mode_frac_hb", "mode_frac_ir",
    "n_unique_ours", "n_unique_ts", "n_unique_hb", "n_unique_ir",
    "conf_gap_ours", "conf_gap_ts", "conf_gap_hb", "conf_gap_ir",
    # n_reports_by_class is a nested dict — stored in JSON only, not CSV
]


def save_outputs(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Strip private arrays before serializing to JSON.
    # n_reports_by_class, ece_sweep, jd_sweep ARE included in JSON (not CSV).
    def strip(r):
        return {k: v for k, v in r.items()
                if not k.startswith("_") and k not in ("_per_bin_ours", "_per_bin_ts")}

    json_path = os.path.join(output_dir, "summary_table.json")
    with open(json_path, "w") as f:
        json.dump([strip(r) for r in rows], f, indent=2)
    print(f"Saved: {json_path}")

    csv_path = os.path.join(output_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: strip(r).get(k, "") for k in CSV_FIELDS})
    print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Failure-mode analysis for MMLU-Redux and MMLU.")
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir",  default="./results/failure_mode_analysis")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS,
                        help="Datasets to analyse (default: MMLU_Redux MMLU).")
    add_selection_arguments(parser)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Collect per-pair results across all datasets ---
    all_rows = []
    for dataset in args.datasets:
        ds_label = DATASET_CONFIG[dataset]["label"]
        for inst_slug, base_slug in selected_models(args):
            sname = SHORT_NAME.get(inst_slug, inst_slug)
            print(f"\n{'='*60}\n[{ds_label}] {sname}")
            row = analyze_pair(args.results_dir, dataset, inst_slug, base_slug)
            if row is None:
                continue
            all_rows.append(row)
            wins_str = "✓ wins" if row["ours_wins"] else ("✗ fails" if row["ours_wins"] is False else "≈ tie")
            print(f"  {wins_str}: ECE_ours={row['ece_ours']:.4f}  "
                  f"ECE_ts={row['ece_ts']:.4f}  ECE_hb={row['ece_hb']:.4f}  "
                  f"ECE_ir={row['ece_ir']:.4f}  delta={row['delta']:+.4f}")
            print(f"  max_conf_bin={row['max_conf_bin_frac']:.3f}  "
                  f"max_report(j,m)={row['max_report_frac']:.3f}  "
                  f"n_r1={row['n_unique_inst_reports']}  "
                  f"base_mean={row['base_mean_conf_calib']:.3f}  "
                  f"TV_conf={row['tv_conf']:.3f}  TV_report={row['tv_report']:.3f}  "
                  f"disc_ours={row['disc_ours']:.3f}")

    if not all_rows:
        print("\nNo successful results. Check data paths.")
        return

    # --- Save tables ---
    save_outputs(all_rows, args.output_dir)

    # --- Aggregate plots (all datasets combined for scatter/histogram) ---
    plot_scatter_grid(all_rows, args.output_dir)
    plot_confidence_histograms(all_rows, args.output_dir)
    plot_per_bin_calibration(all_rows, args.output_dir)

    # Per-dataset reliability grids
    for dataset in args.datasets:
        plot_compact_reliability_grid(all_rows, args.output_dir, dataset=dataset)

    # Sensitivity analysis plots for the focal MMLU_Redux pairs.
    plot_ece_bin_counts_focal(all_rows, args.output_dir)
    plot_ece_sensitivity_focal(all_rows, args.output_dir)
    plot_jd_sensitivity_focal(all_rows, args.output_dir)

    # Detailed plots for focal pairs (MMLU_Redux)
    for inst_slug, tag in [(FOCAL_FAILURE, "LLaMA"), (FOCAL_SUCCESS, "Qwen8B")]:
        focal_rows = [r for r in all_rows
                      if r["inst_slug"] == inst_slug and r["dataset"] == "MMLU_Redux"]
        if focal_rows:
            plot_reliability_4panel(focal_rows[0], args.output_dir, tag)
            plot_recalibration_map(focal_rows[0],  args.output_dir, tag)

    # --- Print summary table ---
    for dataset in args.datasets:
        ds_label = DATASET_CONFIG[dataset]["label"]
        ds_rows  = [r for r in all_rows if r["dataset"] == dataset]
        if not ds_rows:
            continue
        print(f"\n{'='*120}")
        print(f"[{ds_label}]")
        print(f"{'Model':<18} {'δ(ts-ours)':>10} {'δ(hb-ours)':>10} {'δ(ir-ours)':>10} "
              f"{'base_mean':>9} {'max_conf':>8} {'max_rpt':>7} {'n_r1':>5} "
              f"{'TV_conf':>7} {'TV_rpt':>7} {'disc_ours':>9}")
        print("-"*120)
        for r in sorted(ds_rows, key=lambda x: x.get("delta", 0)):
            wins = "✓" if r["ours_wins"] else ("✗" if r["ours_wins"] is False else "~")
            n_r1 = r.get("n_unique_inst_reports", float("nan"))
            def _f(v): return f"{v:>+10.4f}" if not np.isnan(v) else "       N/A"
            print(f"{r['short_name']:<18} {_f(r['delta'])} {_f(r['delta_hb'])} {_f(r['delta_ir'])} "
                  f"{r['base_mean_conf_calib']:>9.3f} "
                  f"{r['max_conf_bin_frac']:>8.3f} "
                  f"{r['max_report_frac']:>7.3f} "
                  f"{int(n_r1) if not np.isnan(n_r1) else 'N/A':>5} "
                  f"{r['tv_conf']:>7.3f} "
                  f"{r['tv_report']:>7.3f} "
                  f"{r['disc_ours']:>9.3f}  {wins}")

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
