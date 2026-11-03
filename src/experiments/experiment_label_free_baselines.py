"""
experiment_label_free_baselines.py

Implements and compares label-free and supervised calibration baselines.

Label-free baselines:
  BCT      — Base Confidence Transfer: keep Instruct answer, use Base model's
              probability for that class as confidence.
  Ensemble — Probability Averaging: average full K-class distributions from
              both models; take argmax and max confidence of the average.

Supervised baselines:
  GPcalib  — Gaussian Process calibration (Wenger et al., AISTATS 2020).
              1D GP mapping: confidence -> calibrated confidence, fit on
              labeled calibration set.
  PBR      — PAC-Bayes Recalibration (Fujisawa & Futami, ICML 2025).
              Shared sparse variational GP with expected binary-vector Brier
              plus alpha/n_fit KL; alpha selected by calibration-only CV.

Protocol (same as our method and paper Appendix F.1):
  - Fit / estimate map on calibration set (no labels for BCT/Ensemble).
  - Apply map to disjoint test set.
  - Report all metrics on test set only.

Data note: All existing .jsonl files already contain `logprobs_raw` with
log-probabilities for every answer choice (A-D or A-E). Full K-class
softmax distributions are derived from these — no pipeline changes needed.

Usage:
    python source/summarize+figures/experiment_label_free_baselines.py \\
        --results_dir ./results \\
        --output_dir  ./results/label_free_baselines
"""

from src.experiments.pbr import result_metadata as pbr_result_metadata
from src.experiments.cli import add_selection_arguments, selected_pairs

from src.experiments.io import load_pair_data, load_jsonl
from src.experiments.settings import SHORT_NAME
from src.config import MAIN_PAIRS
from src.config import DATASET_CONFIG
from src.experiments.baselines import (
    _subsample_for_gp,
    apply_bct,
    apply_conf_avg,
    apply_ensemble,
    apply_gpcalib,
    apply_pbr,
    apply_temperature_scaling,
    fit_daca,
    fit_gpcalib,
    fit_pbr,
    fit_temperature_scaling,
    get_full_dist,
)
from functools import partial
get_full_dist = partial(get_full_dist, allow_uniform=False)
apply_bct = partial(apply_bct, allow_uniform=False)
apply_ensemble = partial(apply_ensemble, allow_uniform=False)
from src.metrics import ece_uwb, is_correct
from src.metrics import compute_metrics

import os
import json
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# GP fitting: subsample calibration data to at most this many samples.
# CommonSenseQA/train has ~9741 samples; GP fitting is O(n^3) so the full
# set would be prohibitively slow.  1500 keeps wall time under ~30s per fit.
# MMLU_Redux uses gp_max_samples=None (no subsampling; n=1531 is manageable).
MAX_GP_CALIB_SAMPLES = 1500

# Per paper Appendix F.1: calibrate on calib set, evaluate on eval set.


# PBR alpha grid: controls KL regularization strength


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _ours_path(results_dir, dataset, split, calib_dataset, calib_split,
               inst_slug, inst_tpl, base_slug, base_tpl):
    return os.path.join(
        results_dir, dataset, split,
        "post_processing", "transformation", "confidence",
        f"calibrated_on_{calib_dataset}_{calib_split}",
        "confidence", "logprob", inst_slug,
        f"{inst_tpl}_{base_slug}_{base_tpl}.jsonl",
    )


def _ts_path(results_dir, dataset, split, inst_slug, inst_tpl, base_slug, base_tpl):
    return os.path.join(
        results_dir, dataset, split,
        "post_processing", "temperature_scaling",
        "confidence", "logprob", inst_slug,
        f"{inst_tpl}_{base_slug}_{base_tpl}.jsonl",
    )


# ---------------------------------------------------------------------------
# Per-pair experiment runner
# ---------------------------------------------------------------------------

def run_pair(dataset, inst_slug, base_slug, results_dir, *, pbr_artifact_dir=None):
    """
    Run all methods for one (dataset, inst_slug, base_slug) pair.
    Returns (metrics_dict, record_sets_dict) or None on failure.
    record_sets_dict maps method name -> list of records (for reliability diagrams).
    GP models are fitted once here and reused — never fitted twice per pair.
    """
    cfg = DATASET_CONFIG[dataset]
    calib_ds    = cfg["calib_dataset"]
    calib_split = cfg["calib_split"]
    apply_ds    = cfg["apply_dataset"]
    apply_split = cfg["apply_split"]
    inst_tpl    = cfg["instruct_tpl"]
    base_tpl    = cfg["base_tpl"]

    # --- Load calibration data ---
    calib_inst, calib_base, test_inst, test_base = load_pair_data(
        results_dir, dataset, inst_slug, base_slug)

    if not test_inst:
        print(f"  [SKIP] Missing test instruct: {apply_ds}/{apply_split}/{inst_slug}")
        return None
    if not test_base:
        print(f"  [SKIP] Missing test base: {apply_ds}/{apply_split}/{base_slug}")
        return None

    results     = {}
    record_sets = {}  # method -> record list (returned for reliability diagrams)

    # -----------------------------------------------------------------------
    # Uncalibrated Instruct (reference point)
    # -----------------------------------------------------------------------
    results["Instruct"] = compute_metrics(test_inst)
    record_sets["Instruct"] = test_inst

    # -----------------------------------------------------------------------
    # Base model (reference point)
    # -----------------------------------------------------------------------
    results["Base"] = compute_metrics(test_base)
    record_sets["Base"] = test_base

    # -----------------------------------------------------------------------
    # Our method — load from existing post-processing output files
    # -----------------------------------------------------------------------
    ours_path = _ours_path(results_dir, apply_ds, apply_split,
                           calib_ds, calib_split,
                           inst_slug, inst_tpl, base_slug, base_tpl)
    ours_records = load_jsonl(ours_path)
    if ours_records:
        results["Ours"] = compute_metrics(ours_records)
        record_sets["Ours"] = ours_records
    else:
        print(f"  [WARN] Ours results not found: {ours_path}")
        results["Ours"] = None

    # -----------------------------------------------------------------------
    # A1: BCT — label-free, no calibration data needed
    # -----------------------------------------------------------------------
    bct_records = apply_bct(test_inst, test_base)
    results["BCT"] = compute_metrics(bct_records)
    record_sets["BCT"] = bct_records

    # -----------------------------------------------------------------------
    # A2: Ensemble — label-free, no calibration data needed
    # -----------------------------------------------------------------------
    ens_records = apply_ensemble(test_inst, test_base)
    results["Ensemble"] = compute_metrics(ens_records)
    record_sets["Ensemble"] = ens_records

    # -----------------------------------------------------------------------
    # A3: ConfAvg — label-free, no calibration data needed
    # -----------------------------------------------------------------------
    confavg_records = apply_conf_avg(test_inst, test_base)
    results["ConfAvg"] = compute_metrics(confavg_records)
    record_sets["ConfAvg"] = confavg_records

    # -----------------------------------------------------------------------
    # DACA — label-free, uses (answer, confidence) scalar from calib set only
    # -----------------------------------------------------------------------
    if calib_inst and calib_base:
        daca_tau = fit_daca(calib_inst, calib_base)
        daca_records = apply_temperature_scaling(daca_tau, test_inst)
        results["DACA"] = compute_metrics(daca_records)
        results["DACA"]["temperature"] = daca_tau
        record_sets["DACA"] = daca_records
    else:
        print(f"  [WARN] Missing calib data — skipping DACA.")

    # -----------------------------------------------------------------------
    # Supervised methods: need labeled calibration set
    # -----------------------------------------------------------------------
    if not calib_inst:
        print(f"  [WARN] Missing calib instruct — skipping supervised methods.")
        return results, record_sets

    # TS — Temperature Scaling uses the full calibration set (scalar opt, fast)
    T = fit_temperature_scaling(calib_inst)
    ts_records = apply_temperature_scaling(T, test_inst)
    results["TS"] = compute_metrics(ts_records)
    results["TS"]["temperature"] = T
    record_sets["TS"] = ts_records

    # Load existing TS output for cross-check (if available)
    ts_existing = load_jsonl(_ts_path(results_dir, apply_ds, apply_split,
                                      inst_slug, inst_tpl, base_slug, base_tpl))
    if ts_existing:
        ts_ece_existing = ece_uwb(ts_existing)
        results["TS"]["ece_existing_file"] = ts_ece_existing

    # GPcalib only: subsample CommonSenseQA (gp_max_samples=1500).
    # MMLU_Redux uses gp_max_samples=None — no subsampling (n=1531 is fine).
    gp_max = cfg.get("gp_max_samples")
    calib_gp = _subsample_for_gp(calib_inst, max_n=gp_max)
    if gp_max is not None and len(calib_inst) > len(calib_gp):
        print(f"  GP calibration: subsampled {len(calib_gp)}/{len(calib_inst)} "
              f"(gp_max_samples={gp_max})", flush=True)
    else:
        print(f"  GP calibration: using all {len(calib_gp)} samples", flush=True)

    # GPcalib — Gaussian Process calibration (Bernoulli likelihood, scalar mode)
    gpc_model = fit_gpcalib(calib_gp)
    gpc_records = apply_gpcalib(gpc_model, test_inst)
    results["GPcalib"] = compute_metrics(gpc_records)
    results["GPcalib"]["gp_calib_n"] = len(calib_gp)
    record_sets["GPcalib"] = gpc_records

    # PBR — shared variational GP, expected Brier + KL, calibration-only CV
    # PBR has its own complete calibration pool; the GP budget is unchanged.
    pbr_expected_n = 9741 if calib_ds == "CommonSenseQA" else 1531
    pbr_model = fit_pbr(calib_inst, expected_n=pbr_expected_n,
                        artifact_dir=pbr_artifact_dir)
    pbr_records = apply_pbr(pbr_model, test_inst)
    results["PBR"] = compute_metrics(pbr_records)
    results["PBR"]["best_alpha"] = pbr_model["best_alpha"]
    results["PBR"].update(pbr_result_metadata(pbr_model))
    record_sets["PBR"] = pbr_records

    return results, record_sets


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

METHOD_ORDER = ["Instruct", "BCT", "Ensemble", "ConfAvg", "DACA", "Ours",
                "TS", "GPcalib", "PBR", "Base"]
METHOD_CATEGORY = {
    "Instruct": "uncalibrated",
    "Base":     "reference",
    "BCT":      "label-free",
    "Ensemble": "label-free",
    "ConfAvg":  "label-free",
    "DACA":     "label-free",
    "Ours":     "label-free",
    "TS":       "supervised",
    "GPcalib":  "supervised",
    "PBR":      "supervised",
}


def print_results_table(all_rows):
    header = f"{'Dataset':<15} {'Model':<18} {'Method':<10} {'Cat':<12} {'ECE':>8} {'Brier':>8} {'Acc':>8} {'CL':>9}"
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("RESULTS: Label-Free and Supervised Baselines (test set only)")
    print(sep)
    print(header)
    print(sep)
    last_pair = None
    for row in all_rows:
        pair = (row["dataset"], row["model"])
        if pair != last_pair and last_pair is not None:
            print(sep)
        last_pair = pair
        ece_str   = f"{row['ece']:.4f}"   if row["ece"]   is not None else "  N/A "
        brier_str = f"{row['brier']:.4f}" if row["brier"] is not None else "  N/A "
        acc_str   = f"{row['acc']:.4f}"   if row["acc"]   is not None else "  N/A "
        cl_str    = f"{row['cl']:.4f}"    if row.get("cl") is not None else "  N/A "
        print(f"{row['dataset']:<15} {row['model']:<18} {row['method']:<10} "
              f"{row['category']:<12} {ece_str:>8} {brier_str:>8} {acc_str:>8} {cl_str:>9}")
    print(sep)


def plot_comparison(all_rows, output_path):
    """Bar-chart comparison of ECE and Brier Score across methods."""
    datasets = list(dict.fromkeys(r["dataset"] for r in all_rows))
    models   = list(dict.fromkeys(r["model"]   for r in all_rows))
    methods  = [m for m in METHOD_ORDER if any(r["method"] == m for r in all_rows)]

    n_ds  = len(datasets)
    n_mod = len(models)
    fig, axes = plt.subplots(2, n_ds * n_mod,
                              figsize=(4.5 * n_ds * n_mod, 8),
                              squeeze=False)

    colors = {
        "Instruct": "#aaaaaa",
        "Base":     "#888888",
        "BCT":      "#1f77b4",
        "Ensemble": "#17becf",
        "ConfAvg":  "#8c564b",
        "DACA":     "#e377c2",
        "Ours":     "#ff7f0e",
        "TS":       "#2ca02c",
        "GPcalib":  "#d62728",
        "PBR":      "#9467bd",
    }
    hatches = {
        "Instruct": "",
        "Base":     "//",
        "BCT":      "",
        "Ensemble": "",
        "ConfAvg":  "",
        "DACA":     "",
        "Ours":     "**",
        "TS":       "//",
        "GPcalib":  "//",
        "PBR":      "//",
    }

    col = 0
    for ds in datasets:
        for mod in models:
            # Gather rows for this (dataset, model)
            pair_rows = {r["method"]: r for r in all_rows
                         if r["dataset"] == ds and r["model"] == mod}
            if not pair_rows:
                col += 1
                continue

            for metric_idx, (metric, ylabel) in enumerate([("ece", "ECE (UWB-10)"),
                                                             ("brier", "Brier Score")]):
                ax = axes[metric_idx][col]
                xs = np.arange(len(methods))
                vals = [pair_rows.get(m, {}).get(metric) for m in methods]
                bars = ax.bar(xs, [v if v is not None else 0.0 for v in vals],
                               color=[colors.get(m, "gray") for m in methods],
                               hatch=[hatches.get(m, "") for m in methods],
                               edgecolor="black", linewidth=0.7)
                # Mark N/A bars
                for i, v in enumerate(vals):
                    if v is None:
                        ax.text(i, 0.002, "N/A", ha="center", va="bottom", fontsize=6)
                ax.set_xticks(xs)
                ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(f"{ds} | {mod}\n{ylabel}", fontsize=8)
                ax.grid(axis="y", alpha=0.3)
            col += 1

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[m], label=f"{m} ({METHOD_CATEGORY[m]})", hatch=hatches[m],
              edgecolor="black")
        for m in methods if m in colors
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Calibration comparison: label-free vs. supervised baselines\n"
        "Hatched = supervised (requires labeled calibration data); "
        "★ = our label-free method",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved → {output_path}")


def plot_reliability_diagrams(all_rows_with_records, output_path):
    """Reliability diagrams for each (dataset, model) pair."""
    methods  = [m for m in METHOD_ORDER if m != "Base"]
    datasets = list(dict.fromkeys(r["dataset"] for r in all_rows_with_records))
    models   = list(dict.fromkeys(r["model"]   for r in all_rows_with_records))

    n_cols = len(methods)
    n_rows = len(datasets) * len(models)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3.5 * n_rows),
                              squeeze=False)
    colors = {"Instruct": "#aaaaaa", "BCT": "#1f77b4", "Ensemble": "#17becf",
              "ConfAvg": "#8c564b", "DACA": "#e377c2", "Ours": "#ff7f0e",
              "TS": "#2ca02c", "GPcalib": "#d62728", "PBR": "#9467bd"}
    row_idx = 0
    for ds in datasets:
        for mod in models:
            pair_rows = {r["method"]: r for r in all_rows_with_records
                         if r["dataset"] == ds and r["model"] == mod}
            if not pair_rows:
                row_idx += 1
                continue
            for col_idx, method in enumerate(methods):
                ax = axes[row_idx][col_idx]
                recs = pair_rows.get(method, {}).get("records")
                if not recs:
                    ax.set_visible(False)
                    continue
                confs   = np.array([r["model_prediction"]["confidence"] for r in recs])
                correct = np.array([float(is_correct(r)) for r in recs])
                bins = np.linspace(0, 1, 11)
                bin_centers, bin_accs = [], []
                for lo, hi in zip(bins[:-1], bins[1:]):
                    mask = (confs > lo) & (confs <= hi) if lo > 0 else (confs <= hi)
                    if mask.sum() > 0:
                        bin_centers.append(confs[mask].mean())
                        bin_accs.append(correct[mask].mean())
                ax.bar(bin_centers, bin_accs, width=0.09,
                       color=colors.get(method, "gray"), alpha=0.7, edgecolor="black", linewidth=0.5)
                ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
                ece = pair_rows.get(method, {}).get("ece", float("nan"))
                ax.set_title(f"{method}\n{ds}/{mod}\nECE={ece:.4f}", fontsize=7)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Confidence", fontsize=7)
                ax.set_ylabel("Accuracy", fontsize=7)
                ax.tick_params(labelsize=6)
            row_idx += 1

    fig.suptitle("Reliability Diagrams — all methods on test set", fontsize=10, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagrams saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Label-free and supervised calibration baselines comparison."
    )
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir",  default="./results/label_free_baselines")
    add_selection_arguments(parser)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows             = []   # for CSV / summary table
    all_rows_with_records = []  # for reliability diagrams (includes record lists)

    for dataset, inst_slug, base_slug in selected_pairs(MAIN_PAIRS, args):
        short = SHORT_NAME.get(inst_slug, inst_slug)
        print(f"\n{'='*60}", flush=True)
        print(f"  {dataset} | {short}", flush=True)
        print(f"{'='*60}", flush=True)

        pair_result = run_pair(
            dataset, inst_slug, base_slug, args.results_dir,
            pbr_artifact_dir=os.path.join(args.output_dir, "artifacts", "PBR", dataset, inst_slug))
        if pair_result is None:
            continue
        results, record_sets = pair_result

        # Collect rows for table/CSV
        for method in METHOD_ORDER:
            m_res = results.get(method)
            if m_res is None:
                continue
            row = {
                "dataset":  dataset,
                "model":    short,
                "instruct": inst_slug,
                "base":     base_slug,
                "method":   method,
                "category": METHOD_CATEGORY.get(method, ""),
                "ece":      m_res.get("ece"),
                "brier":    m_res.get("brier"),
                "acc":      m_res.get("accuracy"),
                "cl":       m_res.get("cl"),
                "n":        m_res.get("n"),
            }
            # Extra fields per method
            if method == "DACA":
                row["daca_tau"] = m_res.get("temperature")
            if method == "TS":
                row["temperature"] = m_res.get("temperature")
            if method == "GPcalib":
                row["gp_calib_n"] = m_res.get("gp_calib_n")
            if method == "PBR":
                row.update({key: value for key, value in m_res.items()
                            if key.startswith("pbr_") or key == "seed"})
            all_rows.append(row)

        # Reliability diagrams: reuse record_sets from run_pair — no re-fitting needed.
        for method, recs in record_sets.items():
            if not recs or method == "Base":
                continue
            all_rows_with_records.append({
                "dataset": dataset,
                "model":   short,
                "method":  method,
                "records": recs,
                "ece":     ece_uwb(recs),
            })

    if not all_rows:
        print("\nNo results collected — check results_dir.")
        return

    # Print summary table
    print_results_table(all_rows)

    # Save CSV
    csv_path = os.path.join(args.output_dir, "comparison_table.csv")
    fieldnames = ["dataset", "model", "method", "category", "ece", "brier",
                  "acc", "cl", "n", "daca_tau", "temperature", "gp_calib_n", "pbr_alpha", "pbr_calib_n", "pbr_objective",
                  "pbr_config_id", "seed", "pbr_training_status", "pbr_calibration_hash",
                  "pbr_model_artifact", "pbr_metadata_artifact"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV saved → {csv_path}")

    # Save JSON
    json_path = os.path.join(args.output_dir, "comparison_table.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"JSON saved → {json_path}")

    # Plots
    plot_comparison(all_rows, os.path.join(args.output_dir, "ece_brier_comparison.pdf"))
    if all_rows_with_records:
        plot_reliability_diagrams(all_rows_with_records,
                                  os.path.join(args.output_dir, "reliability_diagrams.pdf"))

    print("\nDone.")


if __name__ == "__main__":
    main()
