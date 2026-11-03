"""
experiment_all_baselines.py

Unified comparison of all calibration methods: our Bregman-projection method
(real and dummy references), label-free baselines, and supervised baselines.

Methods:
  Uncalibrated:  Instruct, Base
  Label-free:    Ours (real Base), BCT, Ensemble, ConfAvg, DACA,
                 Ours_RandomRef, Ours_CalibratedRef, Ours_ShuffledRandomRef
  Supervised:    TS, GPcalib, PBR

Datasets:  MMLU, MMLU_Redux, CommonSenseQA
Models:    Qwen3-8B, LLaMA-3.1-8B, Ministral-3-8B (three 8B pairs)
Metrics:   Accuracy, Brier Score, ECE (UWB B=10), Confidence Loss

Usage:
    python source/summarize+figures/experiment_all_baselines.py \\
        --results_dir ./results \\
        --output_dir  ./results/all_baselines

Single-pair mode (full GPcalib/PBR fit and CV budgets; not a unit-test shortcut):
    python source/summarize+figures/experiment_all_baselines.py --fast
"""

from src.experiments.pbr import result_metadata as pbr_result_metadata
from src.experiments.cli import add_selection_arguments, selected_pairs

from src.experiments.io import load_pair_data, _conf_path, _ours_output_path, load_jsonl, save_jsonl
from src.experiments.settings import SHORT_NAME
from src.config import MAIN_PAIRS
from src.config import DATASET_CONFIG
from src.experiments.baselines import (
    _subsample_for_gp,
    apply_bct,
    apply_conf_avg,
    apply_ensemble,
    apply_gpcalib,
    apply_histogram_binning,
    apply_isotonic_regression,
    apply_pbr,
    apply_temperature_scaling,
    fit_daca,
    fit_gpcalib,
    fit_histogram_binning,
    fit_isotonic_regression,
    fit_pbr,
    fit_temperature_scaling,
)
from src.experiments.references import (
    DUMMY_RANDOM_SLUG,
    DUMMY_SHUFFLED_RANDOM_SLUG,
    DUMMY_TPL,
    dummy_calibrated_slug,
    make_calibrated_ref,
    make_random_ref,
    make_shuffled_instruct,
    make_shuffled_random_ref,
    shuffled_inst_slug,
)
from src.experiments.pipeline import run_post_processing
from src.metrics import is_correct
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


FAST_PAIRS = [("MMLU_Redux", "Qwen_Qwen3-8B", "Qwen_Qwen3-8B-Base")]


MAX_GP_CALIB_SAMPLES = 1500


METHOD_ORDER = [
    "Instruct", "Base",
    "Ours", "BCT", "Ensemble", "ConfAvg", "DACA",
    "Ours_RandomRef", "Ours_CalibratedRef", "Ours_ShuffledRandomRef",
    "TS", "HB", "IR", "GPcalib", "PBR",
]

METHOD_CATEGORY = {
    "Instruct":               "uncalibrated",
    "Base":                   "reference",
    "Ours":                   "label-free",
    "BCT":                    "label-free",
    "Ensemble":               "label-free",
    "ConfAvg":                "label-free",
    "DACA":                   "label-free",
    "Ours_RandomRef":         "label-free",
    "Ours_CalibratedRef":     "label-free",
    "Ours_ShuffledRandomRef": "label-free",
    "TS":                     "supervised",
    "HB":                     "supervised",
    "IR":                     "supervised",
    "GPcalib":                "supervised",
    "PBR":                    "supervised",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _baseline_pred_path(output_dir, dataset, inst_slug, method):
    """Path for saving per-method prediction records."""
    return os.path.join(output_dir, "predictions", dataset, inst_slug, f"{method}.jsonl")


# ---------------------------------------------------------------------------
# Per-pair experiment runner
# ---------------------------------------------------------------------------

def run_pair(dataset, inst_slug, base_slug, results_dir, *, pbr_artifact_dir=None):
    """
    Run all methods for one (dataset, inst_slug, base_slug) pair.
    Returns (results_dict, record_sets_dict) or None on missing data.
    """
    cfg         = DATASET_CONFIG[dataset]
    calib_ds    = cfg["calib_dataset"]
    calib_split = cfg["calib_split"]
    apply_ds    = cfg["apply_dataset"]
    apply_split = cfg["apply_split"]
    inst_tpl    = cfg["instruct_tpl"]
    base_tpl    = cfg["base_tpl"]
    gp_max      = cfg.get("gp_max_samples")

    calib_inst, calib_base, test_inst, test_base = load_pair_data(
        results_dir, dataset, inst_slug, base_slug)

    if not test_inst:
        print(f"  [SKIP] Missing test instruct: {apply_ds}/{apply_split}/{inst_slug}")
        return None
    if not test_base:
        print(f"  [SKIP] Missing test base: {apply_ds}/{apply_split}/{base_slug}")
        return None

    results, record_sets = {}, {}

    # ------------------------------------------------------------------
    # Uncalibrated
    # ------------------------------------------------------------------
    results["Instruct"]     = compute_metrics(test_inst)
    record_sets["Instruct"] = test_inst

    results["Base"]     = compute_metrics(test_base)
    record_sets["Base"] = test_base

    # ------------------------------------------------------------------
    # Ours (real Base) — load pre-computed Bregman-projection output
    # ------------------------------------------------------------------
    ours_path    = _ours_output_path(results_dir, apply_ds, apply_split, calib_ds, calib_split,
                                     inst_slug, inst_tpl, base_slug, base_tpl)
    ours_records = load_jsonl(ours_path)
    if ours_records:
        results["Ours"]     = compute_metrics(ours_records)
        record_sets["Ours"] = ours_records
    else:
        print(f"  [WARN] Ours (real Base) not found: {ours_path}")
        results["Ours"] = None

    # ------------------------------------------------------------------
    # Label-free baselines (require logprobs_raw in test records)
    # ------------------------------------------------------------------
    bct_recs = apply_bct(test_inst, test_base)
    results["BCT"]     = compute_metrics(bct_recs)
    record_sets["BCT"] = bct_recs

    ens_recs = apply_ensemble(test_inst, test_base)
    results["Ensemble"]     = compute_metrics(ens_recs)
    record_sets["Ensemble"] = ens_recs

    confavg_recs = apply_conf_avg(test_inst, test_base)
    results["ConfAvg"]     = compute_metrics(confavg_recs)
    record_sets["ConfAvg"] = confavg_recs

    if calib_inst and calib_base:
        daca_tau     = fit_daca(calib_inst, calib_base)
        daca_recs    = apply_temperature_scaling(daca_tau, test_inst)
        results["DACA"] = compute_metrics(daca_recs)
        results["DACA"]["temperature"] = daca_tau
        record_sets["DACA"] = daca_recs
    else:
        print(f"  [WARN] Missing calib data — skipping DACA.")

    # ------------------------------------------------------------------
    # Dummy reference variants (Bregman projection with degraded refs)
    # ------------------------------------------------------------------
    if calib_base:
        base_acc_on_calib = float(np.mean([is_correct(r) for r in calib_base]))
        cal_slug          = dummy_calibrated_slug(base_slug)
        sh_inst_sg        = shuffled_inst_slug(inst_slug)

        dummy_random_calib  = make_random_ref(calib_base, seed=42)
        dummy_cal_calib     = make_calibrated_ref(calib_base, calib_base, base_acc_on_calib)
        shuf_ref_calib      = make_shuffled_random_ref(calib_base)
        shuf_calib_inst     = make_shuffled_instruct(calib_inst) if calib_inst else []
        shuf_test_inst      = make_shuffled_instruct(test_inst)

        save_jsonl(dummy_random_calib, _conf_path(results_dir, calib_ds, calib_split, DUMMY_RANDOM_SLUG, DUMMY_TPL))
        save_jsonl(dummy_cal_calib,    _conf_path(results_dir, calib_ds, calib_split, cal_slug, DUMMY_TPL))
        save_jsonl(shuf_ref_calib,     _conf_path(results_dir, calib_ds, calib_split, DUMMY_SHUFFLED_RANDOM_SLUG, DUMMY_TPL))
        if shuf_calib_inst:
            save_jsonl(shuf_calib_inst, _conf_path(results_dir, calib_ds, calib_split, sh_inst_sg, inst_tpl))
        save_jsonl(shuf_test_inst, _conf_path(results_dir, apply_ds, apply_split, sh_inst_sg, inst_tpl))

        print(f"  Running Bregman w/ RandomRef ...")
        run_post_processing(results_dir, calib_ds, calib_split, apply_ds, apply_split,
                            inst_slug, DUMMY_RANDOM_SLUG)

        print(f"  Running Bregman w/ CalibratedRef ...")
        run_post_processing(results_dir, calib_ds, calib_split, apply_ds, apply_split,
                            inst_slug, cal_slug)

        if shuf_calib_inst:
            print(f"  Running Bregman w/ ShuffledRandomRef ...")
            run_post_processing(results_dir, calib_ds, calib_split, apply_ds, apply_split,
                                sh_inst_sg, DUMMY_SHUFFLED_RANDOM_SLUG)

        rand_recs = load_jsonl(_ours_output_path(results_dir, apply_ds, apply_split, calib_ds, calib_split,
                                                  inst_slug, inst_tpl, DUMMY_RANDOM_SLUG, DUMMY_TPL))
        cal_recs  = load_jsonl(_ours_output_path(results_dir, apply_ds, apply_split, calib_ds, calib_split,
                                                  inst_slug, inst_tpl, cal_slug, DUMMY_TPL))
        shuf_recs = load_jsonl(_ours_output_path(results_dir, apply_ds, apply_split, calib_ds, calib_split,
                                                  sh_inst_sg, inst_tpl, DUMMY_SHUFFLED_RANDOM_SLUG, DUMMY_TPL)) \
                    if shuf_calib_inst else []

        for key, recs in [("Ours_RandomRef", rand_recs),
                           ("Ours_CalibratedRef", cal_recs),
                           ("Ours_ShuffledRandomRef", shuf_recs)]:
            if recs:
                results[key]     = compute_metrics(recs)
                record_sets[key] = recs
            else:
                print(f"  [WARN] {key} output not found.")
                results[key] = None
    else:
        print(f"  [WARN] Missing calib base — skipping dummy reference experiments.")

    # ------------------------------------------------------------------
    # Supervised baselines (require labeled calibration set)
    # ------------------------------------------------------------------
    if not calib_inst:
        print(f"  [WARN] Missing calib instruct — skipping supervised baselines.")
        return results, record_sets

    T        = fit_temperature_scaling(calib_inst)
    ts_recs  = apply_temperature_scaling(T, test_inst)
    results["TS"] = compute_metrics(ts_recs)
    results["TS"]["temperature"] = T
    record_sets["TS"] = ts_recs

    hb_model = fit_histogram_binning(calib_inst)
    hb_recs  = apply_histogram_binning(hb_model, test_inst)
    results["HB"]     = compute_metrics(hb_recs)
    record_sets["HB"] = hb_recs

    ir_model = fit_isotonic_regression(calib_inst)
    ir_recs  = apply_isotonic_regression(ir_model, test_inst)
    results["IR"]     = compute_metrics(ir_recs)
    record_sets["IR"] = ir_recs

    calib_gp = _subsample_for_gp(calib_inst, max_n=gp_max)
    if gp_max is not None and len(calib_inst) > len(calib_gp):
        print(f"  GP: subsampled {len(calib_gp)}/{len(calib_inst)} (max={gp_max})")
    else:
        print(f"  GP: using all {len(calib_gp)} samples")

    gpc_model = fit_gpcalib(calib_gp)
    gpc_recs  = apply_gpcalib(gpc_model, test_inst)
    results["GPcalib"] = compute_metrics(gpc_recs)
    results["GPcalib"]["gp_calib_n"] = len(calib_gp)
    record_sets["GPcalib"] = gpc_recs

    # PBR has its own complete calibration pool; the GP budget is unchanged.
    pbr_expected_n = 9741 if calib_ds == "CommonSenseQA" else 1531
    pbr_model = fit_pbr(calib_inst, expected_n=pbr_expected_n,
                        artifact_dir=pbr_artifact_dir)
    pbr_recs  = apply_pbr(pbr_model, test_inst)
    results["PBR"] = compute_metrics(pbr_recs)
    results["PBR"]["best_alpha"]  = pbr_model["best_alpha"]
    results["PBR"].update(pbr_result_metadata(pbr_model))
    record_sets["PBR"] = pbr_recs

    return results, record_sets


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(all_rows):
    header = (f"{'Dataset':<15} {'Model':<18} {'Method':<22} {'Cat':<12} "
              f"{'Acc':>7} {'ECE':>7} {'Brier':>8} {'CL':>8}")
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("RESULTS: All Baselines — test set only")
    print(sep)
    print(header)
    print(sep)

    def _f(v):
        return f"{v:.4f}" if v is not None else "   N/A"

    last_pair = None
    for row in all_rows:
        pair = (row["dataset"], row["model"])
        if pair != last_pair and last_pair is not None:
            print(sep)
        last_pair = pair
        print(f"{row['dataset']:<15} {row['model']:<18} {row['method']:<22} "
              f"{row['category']:<12} {_f(row['acc']):>7} {_f(row['ece']):>7} "
              f"{_f(row['brier']):>8} {_f(row['cl']):>8}")
    print(sep)


def plot_ece_comparison(all_rows, output_path):
    """Bar chart of ECE (UWB-10) for each method grouped by (dataset, model)."""
    methods  = [m for m in METHOD_ORDER if any(r["method"] == m for r in all_rows)]
    datasets = list(dict.fromkeys(r["dataset"] for r in all_rows))
    models   = list(dict.fromkeys(r["model"]   for r in all_rows))

    n_panels = len(datasets) * len(models)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 5), squeeze=False)
    colors = {
        "Instruct": "#aaaaaa", "Base": "#888888",
        "Ours": "#ff7f0e",
        "BCT": "#1f77b4", "Ensemble": "#17becf", "ConfAvg": "#8c564b", "DACA": "#e377c2",
        "Ours_RandomRef": "#7f7f7f", "Ours_CalibratedRef": "#bcbd22", "Ours_ShuffledRandomRef": "#c5b0d5",
        "TS": "#2ca02c", "HB": "#98df8a", "IR": "#ff9896",
        "GPcalib": "#d62728", "PBR": "#9467bd",
    }

    col = 0
    for ds in datasets:
        for mod in models:
            pair_rows = {r["method"]: r for r in all_rows if r["dataset"] == ds and r["model"] == mod}
            if not pair_rows:
                col += 1
                continue
            ax = axes[0][col]
            xs   = np.arange(len(methods))
            vals = [pair_rows.get(m, {}).get("ece") for m in methods]
            ax.bar(xs, [v if v is not None else 0.0 for v in vals],
                   color=[colors.get(m, "gray") for m in methods],
                   edgecolor="black", linewidth=0.6)
            for i, v in enumerate(vals):
                if v is None:
                    ax.text(i, 0.002, "N/A", ha="center", va="bottom", fontsize=5)
            ax.set_xticks(xs)
            ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("ECE (UWB-10)", fontsize=8)
            ax.set_title(f"{ds}\n{mod}", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            col += 1

    fig.suptitle("ECE Comparison — All Baselines (test set)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"ECE bar chart saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified comparison of all calibration baselines."
    )
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--output_dir",  default="./results/all_baselines")
    parser.add_argument("--fast", action="store_true",
                        help="Run one pair with full GP/PBR CV/refit budgets; without filters use MMLU_Redux/Qwen3-8B.")
    add_selection_arguments(parser)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    pairs = selected_pairs(MAIN_PAIRS, args)
    if args.fast:
        pairs = (selected_pairs(FAST_PAIRS, args)
                 if not (args.models or args.datasets) else pairs[:1])

    if args.fast:
        print("[FAST MODE] Running 1 pair only (all methods).")

    all_rows = []

    for dataset, inst_slug, base_slug in pairs:
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

        # Save per-method prediction records
        saved = 0
        for method, recs in record_sets.items():
            if recs:
                pred_path = _baseline_pred_path(args.output_dir, dataset, inst_slug, method)
                save_jsonl(recs, pred_path)
                saved += 1
        print(f"  Saved predictions for {saved} methods → "
              f"{os.path.join(args.output_dir, 'predictions', dataset, inst_slug)}/")

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
                "acc":      m_res.get("accuracy"),
                "ece":      m_res.get("ece"),
                "brier":    m_res.get("brier"),
                "cl":       m_res.get("cl"),
                "n":        m_res.get("n"),
            }
            if method == "DACA":
                row["daca_tau"]    = m_res.get("temperature")
            if method == "TS":
                row["temperature"] = m_res.get("temperature")
            if method == "GPcalib":
                row["gp_calib_n"]  = m_res.get("gp_calib_n")
            if method == "PBR":
                row.update({key: value for key, value in m_res.items()
                            if key.startswith("pbr_") or key == "seed"})
            all_rows.append(row)

    if not all_rows:
        print("\nNo results collected — check --results_dir.")
        return

    print_results_table(all_rows)

    csv_path = os.path.join(args.output_dir, "all_baselines.csv")
    fieldnames = ["dataset", "model", "method", "category",
                  "acc", "ece", "brier", "cl", "n",
                  "daca_tau", "temperature", "gp_calib_n", "pbr_alpha", "pbr_calib_n", "pbr_objective",
                  "pbr_config_id", "seed", "pbr_training_status", "pbr_calibration_hash",
                  "pbr_model_artifact", "pbr_metadata_artifact"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV saved → {csv_path}")

    json_path = os.path.join(args.output_dir, "all_baselines.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"JSON saved → {json_path}")

    plot_ece_comparison(all_rows, os.path.join(args.output_dir, "ece_comparison.pdf"))
    print("\nDone.")


if __name__ == "__main__":
    main()
