"""
experiment_dummy_reference.py

Tests our calibration method's robustness to the quality of the reference model
by replacing the real Base model with two "dummy" reference predictors:

  1. RandomRef — uniform prediction over K choices, random final answer,
                 constant confidence = 1/K (= 0.25 for 4-class tasks).
                 Represents an uninformative reference with no signal.

  2. CalibratedRef — same final answers as the real Base model, but constant
                     confidence = Base model's accuracy on the calibration set.
                     Represents a reference with correct answers but perfectly
                     calibrated (i.e., knows its own average accuracy).

Both dummies are written as .jsonl files under results/ in the standard pipeline
path so that the existing post_processing module can load them unchanged.  The
Bregman-projection calibration (our method) is then applied using each dummy as
the reference, and metrics are reported on the evaluation (test) set only.

Usage:
    python -m source.summarize+figures.experiment_dummy_reference \\
        --results_dir ./results \\
        --output_dir  ./results/dummy_reference_experiment
"""

from src.experiments.cli import add_selection_arguments, selected_pairs

from src.experiments.io import load_pair_data, _conf_path, _ours_output_path, load_jsonl, save_jsonl
from src.experiments.settings import SHORT_NAME
from src.config import MAIN_PAIRS
from src.config import DATASET_CONFIG
from src.experiments.references import (
    DUMMY_RANDOM_SLUG,
    DUMMY_SHUFFLED_RANDOM_SLUG,
    DUMMY_TPL,
    _get_shuffle_perm,
    dummy_calibrated_slug,
    make_calibrated_ref,
    make_random_ref,
    make_shuffled_instruct,
    make_shuffled_random_ref,
    shuffled_inst_slug,
)
from src.experiments.pipeline import run_post_processing
from src.metrics import accuracy, brier_score, ece_uwb, is_correct

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


# Slug names for the dummy reference models (used as directory names under logprob/)
# RandomRef does not depend on the Base model (uniform 0.25 conf, seeded random answers),
# so a single file per dataset/split suffices.

# CalibratedRef confidence = Base model accuracy on calibration set, which varies
# per Base model.  The slug therefore embeds the Base model name so each pair writes
# to its own file and they do not overwrite each other.

# ShuffledRandomRef: choices order is shuffled per question; dummy picks uniformly.
# The shuffle permutation is seeded by question ID only, so the same question always
# gets the same permutation regardless of the model pair.  The dummy file is therefore
# shared per dataset/split (all model pairs produce the same content).

# Shuffled instruct slug: model-specific, because the instruct predictions differ per model.

# Template filename written for dummy files (must contain 'baseline' for load_data filter)

# Bregman-projection precision (paper default: s=0.01)

METHOD_ORDER = ["Instruct", "Ours_RealBase", "Ours_RandomRef", "Ours_CalibratedRef",
                "Ours_ShuffledRandomRef", "Base"]
METHOD_LABEL = {
    "Instruct":               "Instruct (uncalib.)",
    "Ours_RealBase":          "Ours (real Base ref)",
    "Ours_RandomRef":         "Ours (random ref)",
    "Ours_CalibratedRef":     "Ours (calib. ref)",
    "Ours_ShuffledRandomRef": "Ours (shuffled rand ref)",
    "Base":                   "Base (reference)",
}
METHOD_CATEGORY = {
    "Instruct":               "uncalibrated",
    "Ours_RealBase":          "label-free",
    "Ours_RandomRef":         "label-free",
    "Ours_CalibratedRef":     "label-free",
    "Ours_ShuffledRandomRef": "label-free",
    "Base":                   "reference",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _perm_path(results_dir, dataset, split):
    """Path to the per-question shuffle permutations JSON file for ShuffledRandomRef."""
    return os.path.join(results_dir, dataset, split,
                        "LLMs", "confidence", "logprob",
                        DUMMY_SHUFFLED_RANDOM_SLUG, "permutations.json")


# ---------------------------------------------------------------------------
# Shuffled-choices helpers
# ---------------------------------------------------------------------------


def save_permutations(source_records: list, shuffled_records: list, path: str):
    """
    Save per-question shuffle permutations to a JSON file for verification.

    source_records:   original (unshuffled) records — provides original_gold and choices.
    shuffled_records: records after apply_choice_shuffle — provides shuffled_gold.

    Each entry in the output JSON:
      {
        "id":            question id,
        "choices":       original choice list (unchanged by shuffle),
        "original_gold": gold index before shuffling,
        "shuffled_gold": gold index after shuffling (= inv_perm[original_gold]),
        "perm":          list where perm[new_idx] = old_idx
      }

    Verify invariant: perm[shuffled_gold] == original_gold.
    """
    entries = []
    for src, shuf in zip(source_records, shuffled_records):
        qid = src["id"]
        K   = len(src["choices"])
        perm = _get_shuffle_perm(qid, K)
        entries.append({
            "id":            qid,
            "choices":       src["choices"],
            "original_gold": src["gold"],
            "shuffled_gold": shuf["gold"],
            "perm":          perm,
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(records):
    return {
        "ece":      ece_uwb(records),
        "brier":    brier_score(records),
        "accuracy": accuracy(records),
        "n":        len(records),
    }


def compute_ref_diagnostics(records):
    """
    Metrics + confidence-distribution stats for a dummy/reference predictor.
    Useful for verifying the dummy behaves as designed:
      - RandomRef / ShuffledRandomRef: mean_conf ≈ 1/K, accuracy ≈ 1/K
      - CalibratedRef: mean_conf ≈ base_acc, accuracy ≈ base_acc
    Returns None if records is empty.
    """
    if not records:
        return None
    confs = np.array([float(r["model_prediction"]["confidence"]) for r in records])
    diag  = compute_metrics(records)
    diag["mean_conf"] = float(confs.mean())
    diag["std_conf"]  = float(confs.std())
    diag["min_conf"]  = float(confs.min())
    diag["max_conf"]  = float(confs.max())
    return diag


# ---------------------------------------------------------------------------
# Per-pair experiment runner
# ---------------------------------------------------------------------------

def run_pair(dataset, inst_slug, base_slug, results_dir):
    """
    Run the dummy-reference experiment for one (dataset, inst, base) triple.
    Returns dict {method: metrics_dict} or None on missing data.
    """
    cfg         = DATASET_CONFIG[dataset]
    calib_ds    = cfg["calib_dataset"]
    calib_split = cfg["calib_split"]
    apply_ds    = cfg["apply_dataset"]
    apply_split = cfg["apply_split"]
    inst_tpl    = cfg["instruct_tpl"]
    base_tpl    = cfg["base_tpl"]

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    calib_inst, calib_base, test_inst, test_base = load_pair_data(
        results_dir, dataset, inst_slug, base_slug)

    if not test_inst:
        print(f"  [SKIP] Missing test instruct data: {apply_ds}/{apply_split}/{inst_slug}")
        return None
    if not calib_base:
        print(f"  [SKIP] Missing calib base data: {calib_ds}/{calib_split}/{base_slug}")
        return None

    # ------------------------------------------------------------------
    # 2. Compute base accuracy on the calibration set (used by CalibratedRef)
    # ------------------------------------------------------------------
    base_acc_on_calib = float(np.mean([is_correct(r) for r in calib_base]))
    print(f"  Base accuracy on calibration set ({calib_ds}/{calib_split}): "
          f"{base_acc_on_calib:.4f}  (n={len(calib_base)})")

    # ------------------------------------------------------------------
    # 3. Create dummy reference records on the calibration set
    # ------------------------------------------------------------------
    dummy_random_calib     = make_random_ref(calib_base, seed=42)
    dummy_calibrated_calib = make_calibrated_ref(calib_base, calib_base, base_acc_on_calib)

    # CalibratedRef slug embeds the base model name because its confidence value
    # (= Base accuracy on calib set) differs per Base model.
    cal_slug = dummy_calibrated_slug(base_slug)

    # ShuffledRandomRef: per-question choice shuffle applied to both instruct and dummy.
    # Shuffled instruct slug is model-specific; shuffled dummy slug is shared per dataset.
    sh_inst_slug = shuffled_inst_slug(inst_slug)
    shuffled_calib_inst = make_shuffled_instruct(calib_inst) if calib_inst else []
    shuffled_test_inst  = make_shuffled_instruct(test_inst)
    shuffled_random_ref_calib = make_shuffled_random_ref(calib_base)

    # ------------------------------------------------------------------
    # 4. Write all dummy calibration files and shuffled instruct files
    # ------------------------------------------------------------------
    random_calib_path = _conf_path(
        results_dir, calib_ds, calib_split, DUMMY_RANDOM_SLUG, DUMMY_TPL)
    calibrated_calib_path = _conf_path(
        results_dir, calib_ds, calib_split, cal_slug, DUMMY_TPL)
    shuffled_ref_calib_path = _conf_path(
        results_dir, calib_ds, calib_split, DUMMY_SHUFFLED_RANDOM_SLUG, DUMMY_TPL)

    # Shuffled instruct files: written to {inst_slug}_shuffled directory so
    # post_processing can load them as model1 for the ShuffledRandomRef run.
    shuffled_inst_calib_path = _conf_path(
        results_dir, calib_ds, calib_split, sh_inst_slug, inst_tpl)
    shuffled_inst_test_path  = _conf_path(
        results_dir, apply_ds, apply_split, sh_inst_slug, inst_tpl)

    save_jsonl(dummy_random_calib,        random_calib_path)
    save_jsonl(dummy_calibrated_calib,    calibrated_calib_path)
    save_jsonl(shuffled_random_ref_calib, shuffled_ref_calib_path)
    if shuffled_calib_inst:
        save_jsonl(shuffled_calib_inst, shuffled_inst_calib_path)
    save_jsonl(shuffled_test_inst, shuffled_inst_test_path)

    print(f"  Wrote RandomRef calib          → {random_calib_path}")
    print(f"  Wrote CalibratedRef calib      → {calibrated_calib_path}")
    print(f"  Wrote ShuffledRandomRef calib  → {shuffled_ref_calib_path}")
    if shuffled_calib_inst:
        print(f"  Wrote shuffled Instruct calib  → {shuffled_inst_calib_path}")
    print(f"  Wrote shuffled Instruct test   → {shuffled_inst_test_path}")

    # Save per-question shuffle permutations for ShuffledRandomRef verification.
    # File maps each question_id → {choices, original_gold, shuffled_gold, perm}.
    if calib_inst and shuffled_calib_inst:
        perm_calib_path = _perm_path(results_dir, calib_ds, calib_split)
        save_permutations(calib_inst, shuffled_calib_inst, perm_calib_path)
        print(f"  Wrote calib permutations       → {perm_calib_path}")
    perm_test_path = _perm_path(results_dir, apply_ds, apply_split)
    save_permutations(test_inst, shuffled_test_inst, perm_test_path)
    print(f"  Wrote test  permutations       → {perm_test_path}")

    # ------------------------------------------------------------------
    # 4b. Diagnostic metrics for dummy reference records (calibration set)
    #     Checks: RandomRef acc ≈ 1/K, conf constant 1/K
    #             CalibratedRef acc ≈ base_acc, conf constant = base_acc
    #             ShuffledRandomRef acc ≈ 1/K, conf constant 1/K
    # ------------------------------------------------------------------
    diag_random     = compute_ref_diagnostics(dummy_random_calib)
    diag_calibrated = compute_ref_diagnostics(dummy_calibrated_calib)
    diag_shuffled   = compute_ref_diagnostics(shuffled_random_ref_calib)

    print(f"\n  --- Dummy Reference Diagnostics (calibration set, n={len(dummy_random_calib)}) ---")
    print(f"  {'Name':<22}  {'acc':>6}  {'mean_conf':>9}  {'std_conf':>8}  "
          f"{'min_conf':>8}  {'max_conf':>8}  {'ece':>7}  {'brier':>7}")
    for name, diag in [("RandomRef",        diag_random),
                       ("CalibratedRef",     diag_calibrated),
                       ("ShuffledRandomRef", diag_shuffled)]:
        if diag:
            print(f"  {name:<22}  {diag['accuracy']:>6.4f}  {diag['mean_conf']:>9.4f}  "
                  f"{diag['std_conf']:>8.4f}  {diag['min_conf']:>8.4f}  "
                  f"{diag['max_conf']:>8.4f}  {diag['ece']:>7.4f}  {diag['brier']:>7.4f}")

    # ------------------------------------------------------------------
    # 5. Run Bregman-projection with each dummy reference
    # ------------------------------------------------------------------
    print(f"\n  --- Running Ours w/ RandomRef ---")
    run_post_processing(results_dir, calib_ds, calib_split,
                        apply_ds, apply_split, inst_slug, DUMMY_RANDOM_SLUG)

    print(f"\n  --- Running Ours w/ CalibratedRef ({cal_slug}) ---")
    run_post_processing(results_dir, calib_ds, calib_split,
                        apply_ds, apply_split, inst_slug, cal_slug)

    if shuffled_calib_inst:
        print(f"\n  --- Running Ours w/ ShuffledRandomRef (model1={sh_inst_slug}) ---")
        run_post_processing(results_dir, calib_ds, calib_split,
                            apply_ds, apply_split,
                            sh_inst_slug, DUMMY_SHUFFLED_RANDOM_SLUG)
    else:
        print(f"  [WARN] No shuffled calib instruct — skipping ShuffledRandomRef run.")

    # ------------------------------------------------------------------
    # 6. Collect metrics on the evaluation (test/validation) set
    # ------------------------------------------------------------------
    results = {}

    # Instruct uncalibrated
    results["Instruct"] = compute_metrics(test_inst)

    # Real Base reference
    if test_base:
        results["Base"] = compute_metrics(test_base)
    else:
        print(f"  [WARN] Missing test base data — Base metrics unavailable.")
        results["Base"] = None

    # Our method with the real Base reference (pre-existing results)
    real_ours_path = _ours_output_path(
        results_dir, apply_ds, apply_split, calib_ds, calib_split,
        inst_slug, inst_tpl, base_slug, base_tpl)
    real_ours_records = load_jsonl(real_ours_path)
    if real_ours_records:
        results["Ours_RealBase"] = compute_metrics(real_ours_records)
    else:
        print(f"  [WARN] Ours (real Base) output not found: {real_ours_path}")
        results["Ours_RealBase"] = None

    # Our method with RandomRef
    random_ours_path = _ours_output_path(
        results_dir, apply_ds, apply_split, calib_ds, calib_split,
        inst_slug, inst_tpl, DUMMY_RANDOM_SLUG, DUMMY_TPL)
    random_ours_records = load_jsonl(random_ours_path)
    if random_ours_records:
        results["Ours_RandomRef"] = compute_metrics(random_ours_records)
    else:
        print(f"  [WARN] Ours (RandomRef) output not found: {random_ours_path}")
        results["Ours_RandomRef"] = None

    # Our method with CalibratedRef (model-specific slug)
    calibrated_ours_path = _ours_output_path(
        results_dir, apply_ds, apply_split, calib_ds, calib_split,
        inst_slug, inst_tpl, cal_slug, DUMMY_TPL)
    calibrated_ours_records = load_jsonl(calibrated_ours_path)
    if calibrated_ours_records:
        results["Ours_CalibratedRef"] = compute_metrics(calibrated_ours_records)
    else:
        print(f"  [WARN] Ours (CalibratedRef) output not found: {calibrated_ours_path}")
        results["Ours_CalibratedRef"] = None

    # Our method with ShuffledRandomRef
    # Post-processing used sh_inst_slug as model1, so the output lives under sh_inst_slug.
    shuffled_ours_path = _ours_output_path(
        results_dir, apply_ds, apply_split, calib_ds, calib_split,
        sh_inst_slug, inst_tpl, DUMMY_SHUFFLED_RANDOM_SLUG, DUMMY_TPL)
    shuffled_ours_records = load_jsonl(shuffled_ours_path)
    if shuffled_ours_records:
        results["Ours_ShuffledRandomRef"] = compute_metrics(shuffled_ours_records)
    else:
        print(f"  [WARN] Ours (ShuffledRandomRef) output not found: {shuffled_ours_path}")
        results["Ours_ShuffledRandomRef"] = None

    # Attach extra metadata
    for key in ("Ours_RandomRef", "Ours_CalibratedRef", "Ours_ShuffledRandomRef"):
        if results.get(key):
            results[key]["base_acc_on_calib"] = base_acc_on_calib

    # Store diagnostics for dummy references so they can be saved to CSV/JSON.
    results["_dummy_diagnostics"] = {
        "RandomRef_calib":         diag_random,
        "CalibratedRef_calib":     diag_calibrated,
        "ShuffledRandomRef_calib": diag_shuffled,
    }

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results_table(all_rows):
    header = (f"{'Dataset':<15} {'Model':<18} {'Method':<22} "
              f"{'ECE':>8} {'Brier':>8} {'Acc':>8}")
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("RESULTS: Dummy Reference Experiment (evaluation set only)")
    print(sep)
    print(header)
    print(sep)
    last_pair = None
    for row in all_rows:
        pair = (row["dataset"], row["model"])
        if pair != last_pair and last_pair is not None:
            print(sep)
        last_pair = pair
        ece_s   = f"{row['ece']:.4f}"   if row["ece"]   is not None else "  N/A "
        brier_s = f"{row['brier']:.4f}" if row["brier"] is not None else "  N/A "
        acc_s   = f"{row['acc']:.4f}"   if row["acc"]   is not None else "  N/A "
        print(f"{row['dataset']:<15} {row['model']:<18} "
              f"{METHOD_LABEL.get(row['method'], row['method']):<22} "
              f"{ece_s:>8} {brier_s:>8} {acc_s:>8}")
    print(sep)


def plot_results(all_rows, output_path):
    """Bar chart comparing ECE and Brier Score across methods for each model pair."""
    datasets = list(dict.fromkeys(r["dataset"] for r in all_rows))
    models   = list(dict.fromkeys(r["model"]   for r in all_rows))
    methods  = [m for m in METHOD_ORDER if any(r["method"] == m for r in all_rows)]

    n_panels = len(datasets) * len(models)
    fig, axes = plt.subplots(2, n_panels,
                             figsize=(4.5 * n_panels, 8),
                             squeeze=False)

    colors = {
        "Instruct":               "#aaaaaa",
        "Ours_RealBase":          "#ff7f0e",
        "Ours_RandomRef":         "#1f77b4",
        "Ours_CalibratedRef":     "#17becf",
        "Ours_ShuffledRandomRef": "#9467bd",
        "Base":                   "#888888",
    }
    hatches = {
        "Instruct":               "",
        "Ours_RealBase":          "**",
        "Ours_RandomRef":         "",
        "Ours_CalibratedRef":     "",
        "Ours_ShuffledRandomRef": "",
        "Base":                   "//",
    }

    col = 0
    for ds in datasets:
        for mod in models:
            pair_rows = {r["method"]: r for r in all_rows
                         if r["dataset"] == ds and r["model"] == mod}
            if not pair_rows:
                col += 1
                continue
            for metric_idx, (metric, ylabel) in enumerate(
                    [("ece", "ECE (UWB-10)"), ("brier", "Brier Score")]):
                ax = axes[metric_idx][col]
                xs   = np.arange(len(methods))
                vals = [pair_rows.get(m, {}).get(metric) for m in methods]
                ax.bar(xs,
                       [v if v is not None else 0.0 for v in vals],
                       color=[colors.get(m, "gray") for m in methods],
                       hatch=[hatches.get(m, "") for m in methods],
                       edgecolor="black", linewidth=0.7)
                for i, v in enumerate(vals):
                    if v is None:
                        ax.text(i, 0.002, "N/A", ha="center", va="bottom", fontsize=6)
                labels = [METHOD_LABEL.get(m, m) for m in methods]
                ax.set_xticks(xs)
                ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(f"{ds}\n{mod} | {ylabel}", fontsize=8)
                ax.grid(axis="y", alpha=0.3)
            col += 1

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[m], label=METHOD_LABEL.get(m, m),
              hatch=hatches.get(m, ""), edgecolor="black")
        for m in methods if m in colors
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Dummy Reference Experiment\n"
        "Ours with real / random / calibrated-constant reference models",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Dummy reference experiment: test our method with degraded references."
    )
    parser.add_argument("--results_dir", default="./results",
                        help="Root of the results directory tree.")
    parser.add_argument("--output_dir",  default="./results/dummy_reference_experiment",
                        help="Where to write the summary CSV, JSON, and plots.")
    add_selection_arguments(parser)
    # Preserve the original six-pair default; MMLU can be selected explicitly.
    parser.set_defaults(datasets=["MMLU_Redux", "CommonSenseQA"])
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    all_diag_rows = []  # diagnostic rows for dummy references on calibration set

    for dataset, inst_slug, base_slug in selected_pairs(MAIN_PAIRS, args):
        short = SHORT_NAME.get(inst_slug, inst_slug)
        print(f"\n{'='*60}")
        print(f"  {dataset} | {short}")
        print(f"{'='*60}")

        pair_results = run_pair(dataset, inst_slug, base_slug, args.results_dir)
        if pair_results is None:
            continue

        for method in METHOD_ORDER:
            m_res = pair_results.get(method)
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
                "n":        m_res.get("n"),
                "base_acc_on_calib": m_res.get("base_acc_on_calib"),
            }
            all_rows.append(row)

        # Collect dummy reference diagnostics (calibration set)
        dummy_diag = pair_results.get("_dummy_diagnostics", {})
        for ref_key, diag in dummy_diag.items():
            if diag is None:
                continue
            all_diag_rows.append({
                "dataset":    dataset,
                "model":      short,
                "ref":        ref_key,
                "acc":        diag.get("accuracy"),
                "mean_conf":  diag.get("mean_conf"),
                "std_conf":   diag.get("std_conf"),
                "min_conf":   diag.get("min_conf"),
                "max_conf":   diag.get("max_conf"),
                "ece":        diag.get("ece"),
                "brier":      diag.get("brier"),
                "n":          diag.get("n"),
            })

    if not all_rows:
        print("\nNo results collected — check results_dir and that predictions exist.")
        return

    # Print summary table
    print_results_table(all_rows)

    # Save main CSV
    csv_path = os.path.join(args.output_dir, "dummy_reference_results.csv")
    fieldnames = ["dataset", "model", "method", "category",
                  "ece", "brier", "acc", "n", "base_acc_on_calib"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV saved → {csv_path}")

    # Save dummy-reference diagnostics CSV
    if all_diag_rows:
        diag_csv_path = os.path.join(args.output_dir, "dummy_ref_diagnostics.csv")
        diag_fieldnames = ["dataset", "model", "ref",
                           "acc", "mean_conf", "std_conf", "min_conf", "max_conf",
                           "ece", "brier", "n"]
        with open(diag_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=diag_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_diag_rows)
        print(f"Diagnostics CSV saved → {diag_csv_path}")

    # Save JSON (main results + diagnostics combined)
    json_path = os.path.join(args.output_dir, "dummy_reference_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_rows, "dummy_diagnostics": all_diag_rows}, f, indent=2)
    print(f"JSON saved → {json_path}")

    # Bar chart
    plot_results(all_rows, os.path.join(args.output_dir, "dummy_reference_comparison.pdf"))

    print("\nDone.")


if __name__ == "__main__":
    main()
