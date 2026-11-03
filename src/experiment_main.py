"""
experiment_main.py
==================
Evaluate our Bregman-projection calibration method across all model pairs
and datasets. Outputs a CSV, JSON, and comparison plots.

Protocol (Appendix F.1):
  - Fit the calibration map on the calibration split (no labels used).
  - Apply the map to the disjoint test split.
  - Report all metrics on the test split only.

Usage:
    python -m src.experiment_main \\
        --results_dir ./results \\
        --output_dir  ./results/main_experiment
"""

from src.experiments.cli import add_selection_arguments, selected_models

from src.config import SHORT_NAME, DATASET_CONFIG
from src.experiments.io import load_pair_data

import os
import sys
import json
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.metrics import compute_metrics, ece_uwb
from src.calibration import build_joint_dist, fit_bregman, apply_bregman


# ---------------------------------------------------------------------------
# Model pairs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


BREGMAN_BIN_WIDTH = 0.01


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-pair experiment runner
# ---------------------------------------------------------------------------

def run_pair(dataset, inst_slug, base_slug, results_dir, diagnostics_path=None):
    """
    Run Bregman calibration for one (dataset, inst_slug, base_slug) triple.

    Returns (metrics_dict, record_sets_dict) or None if data is missing.
    metrics_dict keys: "Instruct", "Ours"
    """
    cfg = DATASET_CONFIG[dataset]

    calib_inst, calib_base, test_inst, test_base = load_pair_data(
        results_dir, dataset, inst_slug, base_slug)

    if not test_inst:
        print(f"  [SKIP] Missing test instruct: "
              f"{cfg['apply_dataset']}/{cfg['apply_split']}/{inst_slug}", flush=True)
        return None
    if not test_base:
        print(f"  [SKIP] Missing test base: "
              f"{cfg['apply_dataset']}/{cfg['apply_split']}/{base_slug}", flush=True)
        return None
    if not calib_inst or not calib_base:
        print(f"  [SKIP] Missing calibration data for "
              f"{cfg['calib_dataset']}/{cfg['calib_split']}", flush=True)
        return None

    metrics     = {}
    record_sets = {}

    # Uncalibrated instruct (reference point)
    metrics["Instruct"]     = compute_metrics(test_inst)
    record_sets["Instruct"] = test_inst

    # Bregman projection
    print(f"  Building joint distribution "
          f"(bin_width={BREGMAN_BIN_WIDTH}, "
          f"n_calib={len(calib_inst)})...", flush=True)
    joint_dist = build_joint_dist(calib_inst, calib_base, bin_width=BREGMAN_BIN_WIDTH)
    if joint_dist is None:
        print("  [WARN] No aligned calibration pairs — skipping.", flush=True)
        return metrics, record_sets

    print(f"  Grid {joint_dist['P_r1r2'].shape[0]}×"
          f"{joint_dist['P_r1r2'].shape[1]}. Solving minimax...", flush=True)
    diagnostic_args = {"diagnostics_path": diagnostics_path} if diagnostics_path is not None else {}
    r1_map, solver_stats = fit_bregman(joint_dist, **diagnostic_args)

    if r1_map is None:
        print("  [WARN] Bregman solver failed.", flush=True)
        return metrics, record_sets

    ss = solver_stats or {}
    print(f"  Solver: status={ss.get('solver_status')}, "
          f"time={ss.get('solver_time_s', 0):.1f}s, "
          f"vars={ss.get('n_vars')}", flush=True)

    ours_recs             = apply_bregman(r1_map, joint_dist, test_inst,
                                          bin_width=BREGMAN_BIN_WIDTH)
    metrics["Ours"]       = compute_metrics(ours_recs)
    metrics["Ours"]["solver_stats"] = solver_stats
    record_sets["Ours"]   = ours_recs

    return metrics, record_sets


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(all_rows):
    header = ("{:<15} {:<14} {:<10} {:>8} {:>8} {:>8} {:>9}"
              .format("Dataset", "Model", "Method", "ECE", "Brier", "Acc", "CL"))
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("RESULTS — Bregman projection vs. uncalibrated, test split only")
    print(sep)
    print(header)
    print(sep)
    last_pair = None
    for row in all_rows:
        pair = (row["dataset"], row["model"])
        if pair != last_pair and last_pair is not None:
            print(sep)
        last_pair = pair
        fmt = lambda v: f"{v:.4f}" if v is not None else "  N/A "
        print("{:<15} {:<14} {:<10} {:>8} {:>8} {:>8} {:>9}".format(
            row["dataset"], row["model"], row["method"],
            fmt(row["ece"]), fmt(row["brier"]), fmt(row["acc"]), fmt(row["cl"])))
    print(sep)


def plot_ece_improvement(all_rows, output_path):
    """Scatter plot: Instruct ECE (x) vs Ours ECE (y), one point per pair."""
    datasets = list(dict.fromkeys(r["dataset"] for r in all_rows))
    colors   = {"MMLU": "#1f77b4", "MMLU_Redux": "#ff7f0e", "CommonSenseQA": "#2ca02c"}
    markers  = {"MMLU": "o", "MMLU_Redux": "s", "CommonSenseQA": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, metric in enumerate(["ece", "brier"]):
        ax = axes[ax_idx]
        instruct_rows = {(r["dataset"], r["model"]): r
                         for r in all_rows if r["method"] == "Instruct"}
        ours_rows     = {(r["dataset"], r["model"]): r
                         for r in all_rows if r["method"] == "Ours"}
        for ds in datasets:
            xs, ys, labels = [], [], []
            for key, row_i in instruct_rows.items():
                if key[0] != ds:
                    continue
                row_o = ours_rows.get(key)
                if row_o is None:
                    continue
                v_i = row_i.get(metric)
                v_o = row_o.get(metric)
                if v_i is None or v_o is None:
                    continue
                xs.append(v_i)
                ys.append(v_o)
                labels.append(key[1])
            ax.scatter(xs, ys, label=ds, color=colors.get(ds, "gray"),
                       marker=markers.get(ds, "o"), s=60, alpha=0.8)
            for x, y, lbl in zip(xs, ys, labels):
                ax.annotate(lbl, (x, y), textcoords="offset points",
                            xytext=(4, 2), fontsize=5, alpha=0.7)

        lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
        lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0]) * 0.95
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "k--", linewidth=1, alpha=0.5, label="y = x")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel(f"Instruct {metric.upper()}", fontsize=10)
        ax.set_ylabel(f"Ours (Bregman) {metric.upper()}", fontsize=10)
        ax.set_title(f"{metric.upper()}: before vs. after Bregman calibration", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Bregman Projection Calibration — all model pairs", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved → {output_path}", flush=True)


def plot_reliability_diagrams(all_rows_with_records, output_path):
    """2-column reliability diagrams (Instruct | Ours) per (dataset, model) pair."""
    methods  = ["Instruct", "Ours"]
    colors   = {"Instruct": "#aaaaaa", "Ours": "#ff7f0e"}

    pairs = list(dict.fromkeys(
        (r["dataset"], r["model"]) for r in all_rows_with_records))

    n_rows = len(pairs)
    n_cols = len(methods)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    pair_map = {}
    for r in all_rows_with_records:
        pair_map.setdefault((r["dataset"], r["model"]), {})[r["method"]] = r

    for row_idx, (ds, mod) in enumerate(pairs):
        method_rows = pair_map.get((ds, mod), {})
        for col_idx, method in enumerate(methods):
            ax  = axes[row_idx][col_idx]
            row = method_rows.get(method)
            if row is None or not row.get("records"):
                ax.set_visible(False)
                continue
            recs    = row["records"]
            confs   = np.array([r["model_prediction"]["confidence"] for r in recs])
            correct = np.array([float(r["choices"][r["gold"]] ==
                                      r["model_prediction"]["final_answer"])
                                for r in recs])
            bins = np.linspace(0, 1, 11)
            bcs, bas = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (confs <= hi) if lo == 0 else ((confs > lo) & (confs <= hi))
                if mask.sum() > 0:
                    bcs.append(confs[mask].mean())
                    bas.append(correct[mask].mean())
            ax.bar(bcs, bas, width=0.09, color=colors.get(method, "gray"),
                   alpha=0.8, edgecolor="black", linewidth=0.5)
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
            ece = row.get("ece", float("nan"))
            ax.set_title(f"{ds} | {mod}\n{method}  ECE={ece:.4f}", fontsize=7)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence", fontsize=7)
            ax.set_ylabel("Accuracy", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Reliability Diagrams — Instruct vs. Bregman (test split)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Reliability diagrams saved → {output_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bregman-projection calibration experiment — all model pairs."
    )
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir",  default="./results/main_experiment")
    add_selection_arguments(parser)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows              = []
    all_rows_with_records = []

    for dataset in (args.datasets or DATASET_CONFIG):
        for inst_slug, base_slug in selected_models(args):
            short = SHORT_NAME.get(inst_slug, inst_slug)
            print(f"\n{'='*60}", flush=True)
            print(f"  {dataset} | {short}", flush=True)
            print(f"{'='*60}", flush=True)

            diagnostics_path = os.path.join(
                args.output_dir, "diagnostics", dataset, f"{inst_slug}__{base_slug}.json",
            )
            pair_result = run_pair(dataset, inst_slug, base_slug, args.results_dir,
                                   diagnostics_path=diagnostics_path)
            if pair_result is None:
                continue
            metrics, record_sets = pair_result

            for method in ["Instruct", "Ours"]:
                m_res = metrics.get(method)
                if m_res is None:
                    continue
                row = {
                    "dataset":  dataset,
                    "model":    short,
                    "instruct": inst_slug,
                    "base":     base_slug,
                    "method":   method,
                    "ece":      m_res.get("ece"),
                    "brier":    m_res.get("brier"),
                    "acc":      m_res.get("accuracy"),
                    "cl":       m_res.get("cl"),
                    "n":        m_res.get("n"),
                }
                if method == "Ours" and m_res.get("solver_stats"):
                    ss = m_res["solver_stats"]
                    row["solver_status"] = ss.get("solver_status")
                    row["solver_time_s"] = ss.get("solver_time_s")
                    row["n_vars"]        = ss.get("n_vars")
                all_rows.append(row)

            for method, recs in record_sets.items():
                if not recs:
                    continue
                all_rows_with_records.append({
                    "dataset": dataset,
                    "model":   short,
                    "method":  method,
                    "records": recs,
                    "ece":     ece_uwb(recs),
                })

    if not all_rows:
        print("\nNo results collected — check results_dir.", flush=True)
        return

    print_results_table(all_rows)

    csv_path = os.path.join(args.output_dir, "results.csv")
    fieldnames = [
        "dataset", "model", "method",
        "ece", "brier", "acc", "cl", "n",
        "solver_status", "solver_time_s", "n_vars",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV saved → {csv_path}", flush=True)

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [{k: v for k, v in r.items() if k not in ("solver_stats",)}
             for r in all_rows],
            f, indent=2)
    print(f"JSON saved → {json_path}", flush=True)

    plot_ece_improvement(all_rows,
                         os.path.join(args.output_dir, "ece_brier_scatter.pdf"))
    if all_rows_with_records:
        plot_reliability_diagrams(all_rows_with_records,
                                  os.path.join(args.output_dir,
                                               "reliability_diagrams.pdf"))

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
