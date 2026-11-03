"""
Batch runner: sample complexity experiment for all (Base, Instruct) pairs
from the paper's Table 1 experiments.

Single-seed mode (default, n_seeds=1):
  Runs the full analysis for each pair with one seed and produces:
  - Per-pair results JSON at results/sample_complexity/{dataset}/{split}/{pair}/results.json
  - Combined summary table printed to stdout and saved to
    results/sample_complexity/summary.json
  - Combined figure at results/sample_complexity/all_pairs.pdf

Multi-seed mode (n_seeds > 1):
  Runs simple separation with N different seeds per pair to measure N* variability.
  Produces:
  - Per-pair results JSON at results/sample_complexity/{dataset}/{split}/{pair}/multi_seed_results.json
  - Combined summary table with mean ± std of N* across seeds
  - Histogram figure at results/sample_complexity/all_pairs_multi_seed.pdf

Datasets and splits (5 groups × 3 pairs = 15 total):
  - MMLU/validation  — calibration set shared by MMLU and MMLU-Redux (n=1531)
  - MMLU/test        — MMLU evaluation set (n=14042)
  - MMLU_Redux/test  — MMLU-Redux evaluation set (n=5700)
  - CommonSenseQA/train      — calibration set for CSQA (n=9741)
  - CommonSenseQA/validation — evaluation set for CSQA (n=1221)

Usage:
  python -m src.experiments sample-complexity [options]

Options:
  --results_dir    Root results directory (default: ./results)
  --output_dir     Where to save experiment outputs (default: ./results/sample_complexity)
  --n_reps         Repetitions per N per seed (default: 1000)
  --n_seeds        Number of seeds for N* variability (default: 1; set >1 for multi-seed)
  --threshold      Legacy strict-threshold fallback; simple does not use it (default: 0.95)
  --threshold_strict  Strict separation-rate threshold (default: 0.80)
  --seed           Base random seed (default: 42)
  --run_bootstrap  Also run the strict bootstrap CI criterion (slow; off by default)
  --n_bootstrap    Bootstrap samples if --run_bootstrap is set (default: 1000)
  --fast           Alias for --n_reps 200 (quick smoke-test)
"""

from src.experiments.cli import add_selection_arguments, SLUG_FAMILY

from src.experiments.sample_complexity_reference_validation import (
    run_pair,
    find_n_star_simple,
    load_confidence_file,
    extract_arrays,
    calculate_ece,
    run_simple_separation,
)
from src.config import MODEL_PAIRS
from src.experiments.settings import SHORT_NAME

import os
import sys
import json
import argparse

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Import the core module (handles the "+" in directory name)
# ---------------------------------------------------------------------------

# Experiment configuration — (Base, Instruct) pairs from Table 1
# ---------------------------------------------------------------------------

def _pair(dataset, split, pair_name, base_model, inst_model,
          base_file="baseline.jsonl", inst_file="baseline_thinking.jsonl"):
    return dict(dataset=dataset, split=split, pair_name=pair_name,
                base_model=base_model, base_file=base_file,
                inst_model=inst_model, inst_file=inst_file)


GROUP_LABEL = {
    ("MMLU",          "validation"): "MMLU\ncalib set (n=1531)",
    ("MMLU",          "test"):       "MMLU\neval set (n=14042)",
    ("MMLU_Redux",    "test"):       "MMLU-Redux\neval set (n=5700)",
    ("CommonSenseQA", "train"):      "CommonSenseQA\ncalib set (n=9741)",
    ("CommonSenseQA", "validation"): "CommonSenseQA\neval set (n=1221)",
}
GROUP_ORDER = [
    ("MMLU",          "validation"),
    ("MMLU",          "test"),
    ("MMLU_Redux",    "test"),
    ("CommonSenseQA", "train"),
    ("CommonSenseQA", "validation"),
]

PAIRS = [_pair(ds, split, SHORT_NAME[inst], base, inst)
         for ds, split in GROUP_ORDER for inst, base in MODEL_PAIRS]

N_GRID = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150,
          160, 170, 180, 190, 200, 250, 300, 400, 500, 750, 1000]


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_pair_multi_seed(
    base_file: str,
    instruct_file: str,
    label: str,
    n_grid: list,
    n_reps: int,
    n_seeds: int,
    base_seed: int,
    output_dir: str,
) -> dict:
    """
    Run simple separation with n_seeds different seeds for one (base, instruct) pair.
    Returns a dict with the N* distribution across seeds and summary statistics.
    Saves multi_seed_results.json; skips if already complete.
    Checkpoints seed-level progress so interrupted runs resume cleanly.
    """
    json_path = os.path.join(output_dir, "multi_seed_results.json")
    if os.path.exists(json_path):
        print(f"\n[SKIP] Already complete: {json_path}")
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Multi-seed pair : {label}")
    print(f"  n_seeds={n_seeds}  n_reps_per_seed={n_reps}  base_seed={base_seed}")

    base_records = load_confidence_file(base_file)
    inst_records = load_confidence_file(instruct_file)
    base_correct, base_conf = extract_arrays(base_records)
    inst_correct, inst_conf = extract_arrays(inst_records)

    n_total = len(base_correct)
    base_ece_full = calculate_ece(base_correct, base_conf)
    inst_ece_full = calculate_ece(inst_correct, inst_conf)
    ece_gap = inst_ece_full - base_ece_full
    print(f"  N (full) = {n_total}  ECE gap = {ece_gap:+.4f}  "
          f"(base {base_ece_full:.4f} / inst {inst_ece_full:.4f})")

    effective_grid = [n for n in n_grid if n <= n_total] or [n_total]

    # Seed-level checkpoint
    ckpt_path = os.path.join(output_dir, "multi_seed_progress.json")
    n_star_per_seed = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path, encoding="utf-8") as f:
            n_star_per_seed = {int(k): v for k, v in json.load(f).items()}
        print(f"  [resume] {len(n_star_per_seed)}/{n_seeds} seeds already done")

    os.makedirs(output_dir, exist_ok=True)

    for seed_idx in tqdm(range(n_seeds), desc=f"  {label}"):
        if seed_idx in n_star_per_seed:
            continue
        seed = base_seed + seed_idx
        simple_results = run_simple_separation(
            base_correct, base_conf, inst_correct, inst_conf,
            effective_grid, n_reps, seed,
            checkpoint_path=None,
            verbose=False,
        )
        n_star = find_n_star_simple(simple_results)
        n_star_per_seed[seed_idx] = n_star
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(n_star_per_seed, f)

    # Aggregate
    n_star_values = [n_star_per_seed[i] for i in range(n_seeds)]
    n_star_finite = [v for v in n_star_values if v is not None]
    pct_not_found = 1.0 - len(n_star_finite) / n_seeds

    if n_star_finite:
        arr = np.array(n_star_finite, dtype=float)
        stats = {
            "mean":          round(float(np.mean(arr)), 1),
            "std":           round(float(np.std(arr)), 1),
            "median":        float(np.median(arr)),
            "min":           int(np.min(arr)),
            "max":           int(np.max(arr)),
            "p5":            float(np.percentile(arr, 5)),
            "p25":           float(np.percentile(arr, 25)),
            "p75":           float(np.percentile(arr, 75)),
            "p95":           float(np.percentile(arr, 95)),
            "pct_not_found": round(pct_not_found, 4),
        }
    else:
        stats = {"pct_not_found": 1.0}

    record = {
        "label":          label,
        "base_file":      base_file,
        "instruct_file":  instruct_file,
        "n_total":        n_total,
        "base_ece_full":  round(base_ece_full, 6),
        "inst_ece_full":  round(inst_ece_full, 6),
        "full_ece_gap":   round(ece_gap, 6),
        "n_seeds":        n_seeds,
        "n_reps_per_seed": n_reps,
        "n_star_per_seed": n_star_values,
        "n_star_stats":   stats,
        "protocol": {
            "simple_criterion": "smallest N with empirical ECE-gap 2.5th percentile > 0",
            "strict_criterion": "not run in multi-seed mode",
            "outer_subset_repetitions_per_n_per_seed": n_reps,
            "bootstrap_repetitions": 0,
            "seed_count": n_seeds,
            "first_seed": base_seed,
            "seed_rule": "base_seed + seed_index",
            "n_grid_requested": list(n_grid),
            "n_grid_effective": effective_grid,
            "sd_definition": "population SD (ddof=0) over finite N* values only; not a CI or SE",
            "not_reached": "null N* means no tested N satisfied the criterion; excluded from finite statistics",
            "threshold_scope": "threshold arguments do not apply to simple N*",
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved → {json_path}")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return record


# ---------------------------------------------------------------------------
# Multi-seed summary table
# ---------------------------------------------------------------------------

def print_multi_seed_summary_table(all_records: list):
    header = (
        f"{'Pair':<20} {'Dataset/Split':<30} {'ECE gap':>10} "
        f"{'N* mean':>8} {'N* std':>7} {'N* med':>7} "
        f"{'N* min':>7} {'N* max':>7} {'N* [P25,P75]':>14} {'%NF':>6}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("MULTI-SEED SUMMARY: N* variability across random seeds")
    print(f"  (simple N* = first grid N whose empirical ECE-gap 2.5th percentile is > 0)")
    print("  (SD uses finite N* values and ddof=0; %NF = seeds with no satisfying grid point)")
    print(sep)
    print(header)
    print(sep)
    for group_key in GROUP_ORDER:
        group_recs = [r for r in all_records
                      if (r["dataset"], r["split"]) == group_key]
        for r in group_recs:
            st = r.get("n_star_stats", {})
            mean_s  = f"{st['mean']:.1f}"  if "mean"   in st else "N/A"
            std_s   = f"{st['std']:.1f}"   if "std"    in st else "N/A"
            med_s   = f"{st['median']:.0f}" if "median" in st else "N/A"
            min_s   = str(st["min"])        if "min"    in st else "N/A"
            max_s   = str(st["max"])        if "max"    in st else "N/A"
            rng_s   = (f"[{st['p25']:.0f},{st['p75']:.0f}]"
                       if "p25" in st else "N/A")
            pct_s   = f"{st.get('pct_not_found', 1.0):.1%}"
            grp_str = f"{r['dataset']}/{r['split']}"
            print(
                f"{r['pair_name']:<20} {grp_str:<30} "
                f"{r['full_ece_gap']:>10.4f} "
                f"{mean_s:>8} {std_s:>7} {med_s:>7} "
                f"{min_s:>7} {max_s:>7} {rng_s:>14} {pct_s:>6}"
            )
        if group_recs:
            print(sep)
    n_seeds = all_records[0]["n_seeds"] if all_records else "?"
    n_reps  = all_records[0]["n_reps_per_seed"] if all_records else "?"
    print(f"  n_seeds={n_seeds}  n_reps_per_seed={n_reps}\n")


# ---------------------------------------------------------------------------
# Multi-seed combined figure
# ---------------------------------------------------------------------------

def plot_combined_multi_seed(all_records: list, output_path: str):
    """
    3×5 grid of N* histograms (rows = model pairs, columns = dataset/split groups).
    Each cell shows the distribution of N* across seeds, with median and IQR annotated.
    """
    pair_names = list(dict.fromkeys(r["pair_name"] for r in all_records))
    present_groups = [g for g in GROUP_ORDER
                      if any((r["dataset"], r["split"]) == g for r in all_records)]

    nrows = len(pair_names)
    ncols = len(present_groups)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 3.5 * nrows),
                              squeeze=False)

    rec_index = {(r["pair_name"], r["dataset"], r["split"]): r for r in all_records}

    for ri, pair_name in enumerate(pair_names):
        for ci, group_key in enumerate(present_groups):
            ax = axes[ri][ci]
            dataset, split = group_key
            rec = rec_index.get((pair_name, dataset, split))
            if rec is None:
                ax.set_visible(False)
                continue

            values = rec["n_star_per_seed"]
            finite = [v for v in values if v is not None]
            st = rec.get("n_star_stats", {})
            pct_nf = st.get("pct_not_found", 0.0)

            if finite:
                ax.hist(finite, bins=20, color="steelblue", alpha=0.7, edgecolor="white")
                med = st.get("median", np.median(finite))
                ax.axvline(med, color="navy", linestyle="--", linewidth=1.5,
                           label=f"median={med:.0f}")
                if "p25" in st and "p75" in st:
                    ax.axvspan(st["p25"], st["p75"], color="steelblue", alpha=0.15,
                               label=f"IQR [{st['p25']:.0f},{st['p75']:.0f}]")
                ax.legend(fontsize=6, loc="upper right")
            else:
                ax.text(0.5, 0.5, "N* not found\nfor any seed",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)

            col_label = GROUP_LABEL.get(group_key, f"{dataset}/{split}")
            mean_str = f"{st['mean']:.1f}" if "mean" in st else "N/A"
            std_str  = f"{st['std']:.1f}"  if "std"  in st else "N/A"
            ax.set_title(
                f"{pair_name} | {col_label}\n"
                f"ECE gap={rec['full_ece_gap']:.4f}  "
                f"N*={mean_str}±{std_str}  "
                f"(NF={pct_nf:.0%})",
                fontsize=7,
            )
            ax.set_xlabel("N* value", fontsize=7)
            ax.set_ylabel("Count (across seeds)", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25)

    n_seeds = all_records[0]["n_seeds"] if all_records else "?"
    n_reps  = all_records[0]["n_reps_per_seed"] if all_records else "?"
    fig.suptitle(
        f"N* variability across {n_seeds} random seeds  (n_reps={n_reps} per seed)\n"
        "Simple N*: first grid N with empirical ECE-gap 2.5th percentile > 0\n"
        "SD: finite N* values, ddof=0; NF: no satisfying point in the tested grid",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMulti-seed figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Single-seed summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_records: list, threshold: float, threshold_strict: float):
    header = (
        f"{'Pair':<20} {'Dataset/Split':<30} {'ECE gap':>10} "
        f"{'Base ECE':>10} {'Inst ECE':>10} "
        f"{'N* simple':>10} {'N* strict':>10}"
    )
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"SUMMARY: Minimum labeled samples N* to identify Base as better calibrated")
    print(f"         simple: empirical gap 2.5th percentile > 0; strict separation rate >= {threshold_strict:.0%}")
    print(sep)
    print(header)
    print(sep)
    for group_key in GROUP_ORDER:
        group_recs = [r for r in all_records
                      if (r["dataset"], r["split"]) == group_key]
        for r in group_recs:
            n_s = str(r["n_star_simple"]) if r["n_star_simple"] is not None else ">max"
            n_b = (str(r["n_star_strict"]) if r["n_star_strict"] is not None else
                   (">max" if any(value is not None for value in r["strict"].values()) else "not-run"))
            group_str = f"{r['dataset']}/{r['split']}"
            print(
                f"{r['pair_name']:<20} {group_str:<30} "
                f"{r['full_ece_gap']:>10.4f} "
                f"{r['base_ece_full']:>10.4f} "
                f"{r['inst_ece_full']:>10.4f} "
                f"{n_s:>10} {n_b:>10}"
            )
        if group_recs:
            print(sep)
    print(
        "  N* simple: first grid N with empirical ECE-gap 2.5th percentile > 0\n"
        f"  N* strict: min N where 95% bootstrap CIs are non-overlapping in ≥{threshold_strict:.0%} of draws\n"
        "  ECE gap  : ECE_instruct − ECE_base on the stated calibration/evaluation pool\n"
        "  >max means criterion not reached on the grid; skipped strict runs show not-run\n"
        "  --threshold is a legacy strict fallback, not the simple criterion\n"
    )


# ---------------------------------------------------------------------------
# Single-seed combined figure
# ---------------------------------------------------------------------------

def plot_combined(all_records: list, output_path: str, threshold: float):
    pair_names = list(dict.fromkeys(r["pair_name"] for r in all_records))
    present_groups = [g for g in GROUP_ORDER
                      if any((r["dataset"], r["split"]) == g for r in all_records)]

    nrows = len(pair_names)
    ncols = len(present_groups)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4 * nrows),
                              squeeze=False)

    rec_index = {(r["pair_name"], r["dataset"], r["split"]): r for r in all_records}

    for ri, pair_name in enumerate(pair_names):
        for ci, group_key in enumerate(present_groups):
            ax = axes[ri][ci]
            dataset, split = group_key
            rec = rec_index.get((pair_name, dataset, split))
            if rec is None:
                ax.set_visible(False)
                continue

            ns_s = sorted(rec["simple"].keys())
            rs_s = [rec["simple"][n]["separation_rate"] for n in ns_s]
            ci_lo_s = [rec["simple"][n]["ci_lower"] for n in ns_s]
            ci_hi_s = [rec["simple"][n]["ci_upper"] for n in ns_s]
            ax.semilogx(ns_s, rs_s, "o-", color="steelblue", linewidth=1.5,
                        markersize=4, label="Simple")
            ax.fill_between(ns_s, ci_lo_s, ci_hi_s, color="steelblue", alpha=0.15)

            if any(v is not None for v in rec["strict"].values()):
                ns_b = sorted(k for k, v in rec["strict"].items() if v is not None)
                rs_b = [rec["strict"][n]["separation_rate"] for n in ns_b]
                ax.semilogx(ns_b, rs_b, "s--", color="tomato", linewidth=1.5,
                            markersize=4, label="Strict (bootstrap CI)")

            ax.axhline(threshold, color="gray", linestyle=":", linewidth=1,
                       label=f"Legacy reference line ({threshold:.0%}); not simple criterion")

            n_s = rec["n_star_simple"]
            n_b = rec["n_star_strict"]
            if n_s is not None:
                ax.axvline(n_s, color="steelblue", linestyle=":", alpha=0.5)
            if n_b is not None:
                ax.axvline(n_b, color="tomato", linestyle=":", alpha=0.5)

            n_s_str = str(n_s) if n_s is not None else ">max"
            n_b_str = str(n_b) if n_b is not None else ">max"

            col_label = GROUP_LABEL.get(group_key, f"{dataset}/{split}")
            ax.set_title(
                f"{pair_name} | {col_label}\n"
                f"ECE gap = {rec['full_ece_gap']:.4f}  "
                f"(base {rec['base_ece_full']:.4f} / inst {rec['inst_ece_full']:.4f})\n"
                f"N* simple={n_s_str}  N* strict={n_b_str}",
                fontsize=8,
            )
            ax.set_xlabel("Subsample size N (log scale)", fontsize=8)
            ax.set_ylabel("Separation rate", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "Sample complexity for reference calibration validation\n"
        "Fraction of N-sample draws correctly identifying Base as better calibrated (ECE_base < ECE_inst)\n"
        "Pools, left to right when present: MMLU calibration / evaluation; Redux evaluation; CSQA calibration / evaluation",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCombined figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def select_pairs_with_indices(args):
    """Keep the original pair index because it determines the experiment seed."""
    return [(index, cfg) for index, cfg in enumerate(PAIRS)
            if (not args.datasets or cfg["dataset"] in args.datasets)
            and (not args.models or SLUG_FAMILY[cfg["inst_model"]] in args.models)]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch sample complexity experiment for all Table 1 model pairs."
    )
    parser.add_argument("--results_dir", default="./results",
                        help="Root directory containing confidence .jsonl files.")
    parser.add_argument("--output_dir", default="./results/sample_complexity",
                        help="Directory for all experiment outputs.")
    parser.add_argument("--n_reps", type=int, default=1000,
                        help="Outer subsets per N per seed; distinct from bootstrap resamples and number of seeds.")
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of seeds for N* variability measurement. "
                             "Use 1 (default) for single-seed mode. "
                             "Use 100 to measure N* distribution across seeds.")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Legacy fallback for strict N* if threshold_strict is None; simple always uses the empirical gap 2.5th percentile > 0. The CLI strict default is 0.80.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed.")
    parser.add_argument("--run_bootstrap", action="store_true",
                        help="Also run strict bootstrap CI separation in single-seed mode (slow; multi-seed remains simple only).")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Bootstrap resamples per outer rep if --run_bootstrap is set.")
    parser.add_argument("--threshold_strict", type=float, default=0.80,
                        help="Separation rate threshold for N* strict (only used with --run_bootstrap).")
    parser.add_argument("--fast", action="store_true",
                        help="Quick mode: n_reps=200.")
    add_selection_arguments(parser)
    args = parser.parse_args(argv)

    pairs = select_pairs_with_indices(args)

    skip_bootstrap = not args.run_bootstrap

    if args.fast:
        args.n_reps = 200
        print("Fast mode: n_reps=200")

    print(f"Results root : {args.results_dir}")
    print(f"Output dir   : {args.output_dir}")
    print(f"n_reps       : {args.n_reps}")
    print(f"n_seeds      : {args.n_seeds}")
    print(f"threshold    : {args.threshold} (legacy strict fallback; not used by simple)")
    print(f"strict threshold: {args.threshold_strict}")
    print("bootstrap    : " + ("not run (multi-seed is simple only)" if args.n_seeds > 1 else
          (f"yes (n_bootstrap={args.n_bootstrap})" if not skip_bootstrap else "no")))
    print(f"seed         : {args.seed}")

    # -----------------------------------------------------------------------
    # Multi-seed mode
    # -----------------------------------------------------------------------
    if args.n_seeds > 1:
        print(f"\n=== MULTI-SEED MODE: measuring N* variability across {args.n_seeds} seeds ===")
        all_records = []

        for pair_idx, cfg in pairs:
            base_file = os.path.join(
                args.results_dir, cfg["dataset"], cfg["split"], "LLMs",
                "confidence", "logprob", cfg["base_model"], cfg["base_file"],
            )
            inst_file = os.path.join(
                args.results_dir, cfg["dataset"], cfg["split"], "LLMs",
                "confidence", "logprob", cfg["inst_model"], cfg["inst_file"],
            )

            if not os.path.exists(base_file):
                print(f"\n[SKIP] Missing base file: {base_file}")
                continue
            if not os.path.exists(inst_file):
                print(f"\n[SKIP] Missing instruct file: {inst_file}")
                continue

            pair_output_dir = os.path.join(
                args.output_dir, cfg["dataset"], cfg["split"], cfg["pair_name"]
            )

            # Each pair gets a non-overlapping seed block: seed + pair_idx * n_seeds
            pair_base_seed = args.seed + pair_idx * args.n_seeds

            record = run_pair_multi_seed(
                base_file=base_file,
                instruct_file=inst_file,
                label=f"{cfg['pair_name']} | {cfg['dataset']}/{cfg['split']}",
                n_grid=N_GRID,
                n_reps=args.n_reps,
                n_seeds=args.n_seeds,
                base_seed=pair_base_seed,
                output_dir=pair_output_dir,
            )

            record["pair_name"] = cfg["pair_name"]
            record["dataset"]   = cfg["dataset"]
            record["split"]     = cfg["split"]
            record["pool_role"] = "calibration" if (cfg["dataset"], cfg["split"]) in (("MMLU", "validation"), ("CommonSenseQA", "train")) else "evaluation"
            record["global_pair_index"] = pair_idx
            all_records.append(record)

        if not all_records:
            print("\nNo pairs completed.")
            sys.exit(1)

        print_multi_seed_summary_table(all_records)

        summary_path = os.path.join(args.output_dir, "multi_seed_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, indent=2)
        print(f"Multi-seed summary JSON saved → {summary_path}")

        fig_path = os.path.join(args.output_dir, "all_pairs_multi_seed.pdf")
        plot_combined_multi_seed(all_records, fig_path)

        print("\nAll done (multi-seed).")
        return

    # -----------------------------------------------------------------------
    # Single-seed mode (original behaviour)
    # -----------------------------------------------------------------------
    all_records = []

    for pair_idx, cfg in pairs:
        base_file = os.path.join(
            args.results_dir, cfg["dataset"], cfg["split"], "LLMs",
            "confidence", "logprob", cfg["base_model"], cfg["base_file"],
        )
        inst_file = os.path.join(
            args.results_dir, cfg["dataset"], cfg["split"], "LLMs",
            "confidence", "logprob", cfg["inst_model"], cfg["inst_file"],
        )

        if not os.path.exists(base_file):
            print(f"\n[SKIP] Missing base file: {base_file}")
            continue
        if not os.path.exists(inst_file):
            print(f"\n[SKIP] Missing instruct file: {inst_file}")
            continue

        pair_output_dir = os.path.join(
            args.output_dir, cfg["dataset"], cfg["split"], cfg["pair_name"]
        )

        json_path = os.path.join(pair_output_dir, "results.json")
        if os.path.exists(json_path):
            print(f"\n[SKIP] Already complete: {json_path}")
            with open(json_path, encoding="utf-8") as f:
                record = json.load(f)
            record["pair_name"] = cfg["pair_name"]
            record["dataset"]   = cfg["dataset"]
            record["split"]     = cfg["split"]
            record["pool_role"] = "calibration" if (cfg["dataset"], cfg["split"]) in (("MMLU", "validation"), ("CommonSenseQA", "train")) else "evaluation"
            record["global_pair_index"] = pair_idx
            all_records.append(record)
            continue

        record = run_pair(
            base_file=base_file,
            instruct_file=inst_file,
            label=f"{cfg['pair_name']} | {cfg['dataset']}/{cfg['split']}",
            n_grid=N_GRID,
            n_reps=args.n_reps,
            n_bootstrap=args.n_bootstrap,
            threshold=args.threshold,
            seed=args.seed + pair_idx,
            output_dir=pair_output_dir,
            skip_bootstrap=skip_bootstrap,
            threshold_strict=args.threshold_strict,
        )

        record["pair_name"] = cfg["pair_name"]
        record["dataset"]   = cfg["dataset"]
        record["split"]     = cfg["split"]
        record["pool_role"] = "calibration" if (cfg["dataset"], cfg["split"]) in (("MMLU", "validation"), ("CommonSenseQA", "train")) else "evaluation"
        record["global_pair_index"] = pair_idx
        all_records.append(record)

    if not all_records:
        print("\nNo pairs completed. Check that results/ directory is populated.")
        sys.exit(1)

    print_summary_table(all_records, args.threshold, args.threshold_strict)

    summary_path = os.path.join(args.output_dir, "summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"Summary JSON saved → {summary_path}")

    fig_path = os.path.join(args.output_dir, "all_pairs.pdf")
    plot_combined(all_records, fig_path, threshold=args.threshold)

    print("\nAll done.")


if __name__ == "__main__":
    main()
