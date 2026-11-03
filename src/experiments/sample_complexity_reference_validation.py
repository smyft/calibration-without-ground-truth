"""
Sample Complexity Experiment: How many labeled samples are needed to verify
that the Base model is better calibrated than the Instruct model?

Two criteria are measured for each subsample size N:
  - Simple: empirical 2.5th percentile of ECE_instruct - ECE_base > 0
  - Strict: fraction of draws where the 95% bootstrap CI of ECE_base
            lies entirely below the 95% bootstrap CI of ECE_instruct

The simple criterion recomputes 10-bin equal-width ECE on paired subsets
drawn without replacement. The strict criterion is a separate analysis.

Usage (single pair):
  python -m src.experiments reference-validation \
    --base_file      results/MMLU_Redux/test/LLMs/confidence/logprob/Qwen_Qwen3-8B-Base/baseline.jsonl \
    --instruct_file  results/MMLU_Redux/test/LLMs/confidence/logprob/Qwen_Qwen3-8B/baseline_thinking.jsonl \
    --label "Qwen3-8B | MMLU-Redux" \
    --output_dir results/sample_complexity/MMLU_Redux/Qwen3-8B

Usage (batch, all pairs):
  python -m src.experiments sample-complexity
"""


import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_confidence_file(path: str):
    """Load a .jsonl confidence file. Returns list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Confidence file not found: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_arrays(records):
    """
    Extract (is_correct, confidence) numpy arrays from loaded records.
    is_correct: float array of 0/1 values.
    confidence: float array in [0, 1].
    """
    is_correct = np.array(
        [float(r["choices"][r["gold"]] == r["model_prediction"]["final_answer"])
         for r in records],
        dtype=np.float64,
    )
    confidence = np.array(
        [r["model_prediction"]["confidence"] for r in records],
        dtype=np.float64,
    )
    return is_correct, confidence


# ---------------------------------------------------------------------------
# ECE computation
# ---------------------------------------------------------------------------

def calculate_ece(is_correct: np.ndarray, confidences: np.ndarray,
                  n_bins: int = 10) -> float:
    """
    Equal-width binning ECE.
    Consistent with source/summarize+figures/calculate_scores.py.
    """
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(is_correct)
    for i in range(n_bins):
        lo, hi = bin_limits[i], bin_limits[i + 1]
        if i == 0:
            mask = confidences <= hi
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            bin_frac = mask.sum() / n
            ece += abs(confidences[mask].mean() - is_correct[mask].mean()) * bin_frac
    return float(ece)


def adaptive_n_bins(N: int) -> int:
    """
    Scale number of bins down for small subsamples to avoid empty bins
    dominating the ECE estimate. Minimum of 2 bins.
    """
    return max(2, min(10, N // 5))


# ---------------------------------------------------------------------------
# Core experiment: simple separation
# ---------------------------------------------------------------------------

def run_simple_separation(
    base_correct: np.ndarray,
    base_conf: np.ndarray,
    inst_correct: np.ndarray,
    inst_conf: np.ndarray,
    n_grid: list,
    n_reps: int,
    seed: int,
    checkpoint_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    For each N in n_grid, draw n_reps random subsets of size N without
    replacement and record the ECE gap (ECE_inst - ECE_base) for each draw.

    Per N, the following statistics are computed from the n_reps gap samples:
      - separation_rate : fraction of draws where gap > 0
      - mean_gap        : mean of gap distribution
      - ci_lower        : 2.5th percentile of gap distribution
      - ci_upper        : 97.5th percentile of gap distribution

    N* criterion: smallest N where ci_lower > 0, meaning the gap is positive
    at the empirical 2.5th percentile. This is a quantile rule, not a configurable
    separation-rate threshold.

    If checkpoint_path is given, results are written after each N completes
    and previously completed N values are skipped on resume.

    Returns dict {N: {"separation_rate", "mean_gap", "ci_lower", "ci_upper"}}.
    """
    rng = np.random.default_rng(seed)
    n_total = len(base_correct)

    # Resume from checkpoint if available
    results = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            results = {int(k): v for k, v in json.load(f).items()}
        if results:
            print(f"  [resume] simple checkpoint: {sorted(results.keys())} already done")

    for N in tqdm(n_grid, desc="  Simple", leave=False, disable=not verbose):
        N_eff = min(N, n_total)
        if N_eff in results:
            continue

        n_bins = 10
        gaps = np.empty(n_reps)

        for i in range(n_reps):
            idx = rng.choice(n_total, size=N_eff, replace=False)
            ece_base = calculate_ece(base_correct[idx], base_conf[idx], n_bins)
            ece_inst = calculate_ece(inst_correct[idx], inst_conf[idx], n_bins)
            gaps[i] = ece_inst - ece_base

        results[N_eff] = {
            "separation_rate": float(np.mean(gaps > 0)),
            "mean_gap":        float(np.mean(gaps)),
            "ci_lower":        float(np.percentile(gaps, 2.5)),
            "ci_upper":        float(np.percentile(gaps, 97.5)),
        }

        if checkpoint_path:
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f)

    return results


# ---------------------------------------------------------------------------
# Core experiment: strict bootstrap CI separation
# ---------------------------------------------------------------------------

def run_bootstrap_separation(
    base_correct: np.ndarray,
    base_conf: np.ndarray,
    inst_correct: np.ndarray,
    inst_conf: np.ndarray,
    n_grid: list,
    n_reps: int,
    n_bootstrap: int,
    seed: int,
    checkpoint_path: str = None,
) -> dict:
    """
    For each N in n_grid, draw n_reps random subsets. Within each subset,
    compute bootstrap 95% CIs for ECE_base and ECE_instruct. Record whether
    CI_base_upper < CI_inst_lower (strict non-overlapping separation).

    Per N, the following statistics are computed across n_reps outer draws:
      - separation_rate    : fraction of draws where the 95% CIs are non-overlapping
      - ece_base_ci_lower  : mean of per-rep bootstrap 2.5th percentile of ECE_base
      - ece_base_ci_upper  : mean of per-rep bootstrap 97.5th percentile of ECE_base
      - ece_inst_ci_lower  : mean of per-rep bootstrap 2.5th percentile of ECE_inst
      - ece_inst_ci_upper  : mean of per-rep bootstrap 97.5th percentile of ECE_inst

    If checkpoint_path is given, results are written after each N completes
    and previously completed N values are skipped on resume.

    Returns dict {N: {"separation_rate",
                       "ece_base_ci_lower", "ece_base_ci_upper",
                       "ece_inst_ci_lower", "ece_inst_ci_upper"}}.
    The ECE CIs are averaged 95% bootstrap confidence intervals across all
    outer repetitions.
    """
    rng = np.random.default_rng(seed)
    n_total = len(base_correct)

    # Resume from checkpoint if available
    results = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            results = {int(k): v for k, v in json.load(f).items()}
        if results:
            print(f"  [resume] bootstrap checkpoint: {sorted(results.keys())} already done")

    for N in tqdm(n_grid, desc="  Bootstrap", leave=False):
        N_eff = min(N, n_total)
        if N_eff in results:
            continue

        n_bins = 10
        flags = np.zeros(n_reps, dtype=np.float64)
        base_ci_lowers = np.empty(n_reps)
        base_ci_uppers = np.empty(n_reps)
        inst_ci_lowers = np.empty(n_reps)
        inst_ci_uppers = np.empty(n_reps)

        for i in range(n_reps):
            idx = rng.choice(n_total, size=N_eff, replace=False)

            base_sub_c = base_correct[idx]
            base_sub_p = base_conf[idx]
            inst_sub_c = inst_correct[idx]
            inst_sub_p = inst_conf[idx]

            ece_base_boot = np.empty(n_bootstrap)
            ece_inst_boot = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                bi = rng.integers(0, N_eff, size=N_eff)
                ece_base_boot[b] = calculate_ece(base_sub_c[bi], base_sub_p[bi], n_bins)
                ece_inst_boot[b] = calculate_ece(inst_sub_c[bi], inst_sub_p[bi], n_bins)

            base_ci_lowers[i] = float(np.percentile(ece_base_boot, 2.5))
            base_ci_uppers[i] = float(np.percentile(ece_base_boot, 97.5))
            inst_ci_lowers[i] = float(np.percentile(ece_inst_boot, 2.5))
            inst_ci_uppers[i] = float(np.percentile(ece_inst_boot, 97.5))

            if base_ci_uppers[i] < inst_ci_lowers[i]:
                flags[i] = 1.0

        results[N_eff] = {
            "separation_rate":    float(np.mean(flags)),
            "ece_base_ci_lower":  float(np.mean(base_ci_lowers)),
            "ece_base_ci_upper":  float(np.mean(base_ci_uppers)),
            "ece_inst_ci_lower":  float(np.mean(inst_ci_lowers)),
            "ece_inst_ci_upper":  float(np.mean(inst_ci_uppers)),
        }

        if checkpoint_path:
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(results, f)

    return results


# ---------------------------------------------------------------------------
# N* finder
# ---------------------------------------------------------------------------

def find_n_star_simple(separation_dict: dict):
    """
    N* for the simple criterion: smallest N where the 2.5th percentile of the
    ECE gap distribution (ci_lower) is strictly positive.
    Uses the existing np.percentile interpolation; no threshold argument applies.
    Returns None if no such N exists in the grid.
    """
    for N in sorted(separation_dict.keys()):
        entry = separation_dict[N]
        if entry["ci_lower"] > 0:
            return N
    return None


def find_n_star_strict(separation_dict: dict, threshold: float = 0.80):
    """
    N* for the strict bootstrap criterion: smallest N where the separation_rate
    (fraction of draws with non-overlapping 95% CIs) meets the threshold.
    Returns None if no such N exists in the grid.
    """
    for N in sorted(separation_dict.keys()):
        entry = separation_dict[N]
        if entry is not None and entry["separation_rate"] >= threshold:
            return N
    return None


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_separation_curves(
    records: list,
    output_path: str,
    threshold: float = 0.95,
):
    """
    records: list of dicts with keys:
        label, full_ece_gap, simple, strict
        where simple/strict are {N: rate} dicts.
    """
    n_pairs = len(records)
    ncols = min(3, n_pairs)
    nrows = (n_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows),
                              squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    for ax, rec in zip(axes_flat, records):
        ns_simple = sorted(rec["simple"].keys())
        ns_strict = sorted(rec["strict"].keys())
        rates_simple = [rec["simple"][n]["separation_rate"] for n in ns_simple]
        ci_lo_simple = [rec["simple"][n]["ci_lower"] for n in ns_simple]
        ci_hi_simple = [rec["simple"][n]["ci_upper"] for n in ns_simple]
        ns_strict_valid = [n for n in ns_strict if rec["strict"][n] is not None]
        rates_strict = [rec["strict"][n]["separation_rate"] for n in ns_strict_valid]

        ax.semilogx(ns_simple, rates_simple, "o-", color="steelblue",
                    label="Simple (ECE_inst − ECE_base gap CI)")
        ax.fill_between(ns_simple, ci_lo_simple, ci_hi_simple,
                        color="steelblue", alpha=0.15)
        ax.semilogx(ns_strict_valid, rates_strict, "s--", color="tomato",
                    label="Strict (non-overlapping 95% CI)")
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.2,
                   label=f"Legacy --threshold reference = {threshold:.0%}")

        n_star_s = find_n_star_simple(rec["simple"])
        n_star_b = find_n_star_strict(rec["strict"], threshold)

        title_lines = [rec["label"],
                       f"ECE gap = {rec['full_ece_gap']:.4f}  (base {rec['base_ece_full']:.4f} vs inst {rec['inst_ece_full']:.4f})"]
        if n_star_s is not None:
            title_lines.append(f"N* simple = {n_star_s}")
        else:
            title_lines.append("N* simple = >max_N")
        if n_star_b is not None:
            title_lines.append(f"N* strict = {n_star_b}")
        else:
            title_lines.append("N* strict = >max_N")

        ax.set_title("\n".join(title_lines), fontsize=8)
        ax.set_xlabel("Subsample size N")
        ax.set_ylabel("Separation rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    # Hide unused axes
    for ax in axes_flat[n_pairs:]:
        ax.set_visible(False)

    fig.suptitle(
        "Sample complexity for Base vs. Instruct calibration verification\n"
        "(fraction of N-sample draws correctly identifying Base as better calibrated)",
        fontsize=10,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Per-pair runner (used by both CLI and batch script)
# ---------------------------------------------------------------------------

def run_pair(
    base_file: str,
    instruct_file: str,
    label: str,
    n_grid: list,
    n_reps: int,
    n_bootstrap: int,
    threshold: float,
    seed: int,
    output_dir: str,
    skip_bootstrap: bool = False,
    threshold_strict: float = None,
) -> dict:
    """
    Full analysis for a single (base, instruct) file pair.
    Saves per-pair JSON to output_dir and returns a result record.
    """
    print(f"\n{'='*60}")
    print(f"  Pair: {label}")
    print(f"  Base:    {base_file}")
    print(f"  Instruct:{instruct_file}")

    base_records = load_confidence_file(base_file)
    inst_records = load_confidence_file(instruct_file)

    assert len(base_records) == len(inst_records), (
        f"Length mismatch: base={len(base_records)}, instruct={len(inst_records)}"
    )
    # Verify ID alignment
    for i in range(min(5, len(base_records))):
        assert base_records[i]["id"] == inst_records[i]["id"], (
            f"ID mismatch at index {i}: {base_records[i]['id']} vs {inst_records[i]['id']}"
        )

    base_correct, base_conf = extract_arrays(base_records)
    inst_correct, inst_conf = extract_arrays(inst_records)

    n_total = len(base_correct)
    print(f"  N (full set) = {n_total}")

    # Full-set ECE (sanity check against Table 1)
    base_ece_full = calculate_ece(base_correct, base_conf, n_bins=10)
    inst_ece_full = calculate_ece(inst_correct, inst_conf, n_bins=10)
    ece_gap = inst_ece_full - base_ece_full
    print(f"  ECE Base     = {base_ece_full:.4f}")
    print(f"  ECE Instruct = {inst_ece_full:.4f}  (gap = {ece_gap:+.4f})")

    # Cap n_grid at available data
    effective_grid = [n for n in n_grid if n <= n_total]
    if not effective_grid:
        effective_grid = [n_total]

    ckpt_simple    = os.path.join(output_dir, "checkpoint_simple.json")
    ckpt_bootstrap = os.path.join(output_dir, "checkpoint_bootstrap.json")

    print(f"  Running simple separation (n_reps={n_reps}) ...")
    simple_results = run_simple_separation(
        base_correct, base_conf, inst_correct, inst_conf,
        effective_grid, n_reps, seed,
        checkpoint_path=ckpt_simple,
    )

    if not skip_bootstrap:
        print(f"  Running bootstrap separation (n_reps={n_reps}, n_bootstrap={n_bootstrap}) ...")
        strict_results = run_bootstrap_separation(
            base_correct, base_conf, inst_correct, inst_conf,
            effective_grid, n_reps, n_bootstrap, seed + 1,
            checkpoint_path=ckpt_bootstrap,
        )
    else:
        strict_results = {n: None for n in effective_grid}

    thr_strict = threshold_strict if threshold_strict is not None else threshold
    n_star_simple = find_n_star_simple(simple_results)
    n_star_strict = find_n_star_strict(strict_results, thr_strict) if not skip_bootstrap else None

    print(f"  N* simple = {n_star_simple}  (criterion: ci_lower > 0)")
    print(f"  N* strict = {n_star_strict}  (criterion: separation_rate >= {thr_strict:.0%})")

    def _round_simple(entry):
        return {
            "separation_rate": round(entry["separation_rate"], 4),
            "mean_gap":        round(entry["mean_gap"], 6),
            "ci_lower":        round(entry["ci_lower"], 6),
            "ci_upper":        round(entry["ci_upper"], 6),
        }

    def _round_strict(entry):
        if entry is None:
            return None
        return {
            "separation_rate":   round(entry["separation_rate"], 4),
            "ece_base_ci_lower": round(entry["ece_base_ci_lower"], 6),
            "ece_base_ci_upper": round(entry["ece_base_ci_upper"], 6),
            "ece_inst_ci_lower": round(entry["ece_inst_ci_lower"], 6),
            "ece_inst_ci_upper": round(entry["ece_inst_ci_upper"], 6),
        }

    record = {
        "label": label,
        "base_file": base_file,
        "instruct_file": instruct_file,
        "n_total": n_total,
        "base_ece_full": round(base_ece_full, 6),
        "inst_ece_full": round(inst_ece_full, 6),
        "full_ece_gap": round(ece_gap, 6),
        "n_star_simple": n_star_simple,
        "n_star_strict": n_star_strict,
        "threshold_simple_criterion": "ci_lower > 0",
        "threshold_strict": thr_strict,
        "protocol": {
            "simple_criterion": "smallest N with empirical ECE-gap 2.5th percentile > 0",
            "strict_criterion": "fraction of outer subsets with bootstrap upper(Base ECE) < lower(Instruct ECE) >= threshold_strict",
            "outer_subset_repetitions_per_n": n_reps,
            "bootstrap_repetitions_per_subset": 0 if skip_bootstrap else n_bootstrap,
            "configured_bootstrap_repetitions": n_bootstrap,
            "bootstrap_percentiles": [2.5, 97.5],
            "seed_count": 1,
            "simple_seed": seed,
            "strict_seed": None if skip_bootstrap else seed + 1,
            "n_grid_requested": list(n_grid),
            "n_grid_effective": effective_grid,
            "threshold_argument": threshold,
            "threshold_scope": "legacy strict fallback only when threshold_strict is None; never simple",
            "simple_status": "reached" if n_star_simple is not None else "not_reached",
            "strict_status": "not_run" if skip_bootstrap else ("reached" if n_star_strict is not None else "not_reached"),
            "not_reached": "no tested N satisfies the criterion; null is not an imputed finite budget",
            "pool_definition": "the supplied prediction-file split; calibration and evaluation pools are separate diagnostics",
        },
        "simple": {int(k): _round_simple(v) for k, v in simple_results.items()},
        "strict": {int(k): _round_strict(v) for k, v in strict_results.items()},
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"  Results saved → {json_path}")

    # Clean up intermediate checkpoints now that results.json is complete
    for ckpt in (ckpt_simple, ckpt_bootstrap):
        if os.path.exists(ckpt):
            os.remove(ckpt)

    return record


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sample complexity experiment for reference calibration validation."
    )
    parser.add_argument("--base_file", required=True,
                        help="Path to Base model confidence .jsonl file.")
    parser.add_argument("--instruct_file", required=True,
                        help="Path to Instruct model confidence .jsonl file.")
    parser.add_argument("--label", default="",
                        help="Human-readable label for this pair (used in plots).")
    parser.add_argument("--output_dir", default="results/sample_complexity/run",
                        help="Directory to save results JSON and figure.")
    parser.add_argument("--n_grid", nargs="+", type=int,
                        default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250,
                                 300, 400, 500, 750, 1000],
                        help="List of subsample sizes to evaluate.")
    parser.add_argument("--n_reps", type=int, default=1000,
                        help="Number of outer subset repetitions per N; separate from within-subset bootstrap repetitions.")
    parser.add_argument("--n_bootstrap", type=int, default=500,
                        help="Bootstrap resamples for strict CI criterion.")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Legacy strict separation-rate threshold, used only when --threshold_strict is omitted; simple uses empirical gap 2.5th percentile > 0.")
    parser.add_argument("--threshold_strict", type=float, default=None,
                        help="Override the legacy --threshold for strict bootstrap separation only. "
                             "Defaults to --threshold if not set. "
                             "Recommended: 0.80 (strict CI separation is inherently harder).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--skip_bootstrap", action="store_true",
                        help="Skip the bootstrap CI criterion (faster).")
    args = parser.parse_args(argv)

    record = run_pair(
        base_file=args.base_file,
        instruct_file=args.instruct_file,
        label=args.label or os.path.basename(args.output_dir),
        n_grid=args.n_grid,
        n_reps=args.n_reps,
        n_bootstrap=args.n_bootstrap,
        threshold=args.threshold,
        seed=args.seed,
        output_dir=args.output_dir,
        skip_bootstrap=args.skip_bootstrap,
        threshold_strict=args.threshold_strict,
    )

    figure_path = os.path.join(args.output_dir, "separation_curve.pdf")
    plot_separation_curves([record], figure_path, threshold=args.threshold)

    print("\nDone.")
    print(f"  N* simple : {record['n_star_simple']}")
    print(f"  N* strict : {record['n_star_strict']}")


if __name__ == "__main__":
    main()
