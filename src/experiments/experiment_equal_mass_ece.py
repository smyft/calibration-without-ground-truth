"""
experiment_equal_mass_ece.py

Comparison of fixed-width and equal-mass ECE estimators.

Computes six ECE estimators for every (Base, Instruct, Ours, TS, HB, IR, MaxConf)
confidence file found in the results directory for MMLU_Redux, MMLU, and CommonSenseQA:

  UWB-10   fixed-width B=10  (our paper / Guo et al. 2017)
  UWB-15   fixed-width B=15
  UWB-opt  fixed-width B=floor(n^{1/3})  [Paper 2, Fujisawa & Futami ICML 2025]
  UMB-10   equal-mass  B=10
  UMB-15   equal-mass  B=15
  UMB-opt  equal-mass  B=floor(n^{1/3})  [Paper 1, Futami & Fujisawa NeurIPS 2024]

Usage:
    python source/summarize+figures/experiment_equal_mass_ece.py \
        --results_dir ./results \
        --output_dir  ./results/equal_mass_ece
"""

from src.experiments.cli import add_selection_arguments, SLUG_FAMILY

from src.config import DATASET_CONFIG, MODEL_PAIRS

import os
import json
import argparse
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# ECE estimators
# ---------------------------------------------------------------------------

def calculate_ece_uwb(is_correct: np.ndarray,
                      confidences: np.ndarray,
                      n_bins: int = 10) -> float:
    """Fixed-width (UWB) ECE. Matches calculate_scores.py convention exactly."""
    is_correct  = np.asarray(is_correct,  dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    bin_limits  = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bin_limits[:-1], bin_limits[1:])):
        if i == 0:
            mask = confidences <= hi
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_frac = mask.sum() / len(confidences)
        ece += abs(confidences[mask].mean() - is_correct[mask].mean()) * bin_frac
    return float(ece)


def calculate_ece_umb(is_correct: np.ndarray,
                      confidences: np.ndarray,
                      n_bins: int = 10) -> float:
    """Equal-mass (UMB / AdaptiveECE). Each bin gets floor(n/n_bins) samples;
    last bin absorbs any remainder."""
    is_correct  = np.asarray(is_correct,  dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    n = len(confidences)
    if n == 0:
        return 0.0
    order     = np.argsort(confidences)
    s_conf    = confidences[order]
    s_correct = is_correct[order]
    ece = 0.0
    for idx in np.array_split(np.arange(n), n_bins):
        if len(idx) == 0:
            continue
        bin_frac = len(idx) / n
        ece += abs(s_conf[idx].mean() - s_correct[idx].mean()) * bin_frac
    return float(ece)


def optimal_n_bins(n: int) -> int:
    """B* = floor(n^{1/3}) per Futami & Fujisawa (2024) and Fujisawa & Futami (2025)."""
    return max(1, int(np.floor(n ** (1.0 / 3.0))))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_confidence_file(path: str):
    """Returns (is_correct, confidences) as numpy arrays, or (None, None)."""
    if not os.path.exists(path):
        return None, None
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return None, None
    is_correct  = np.array([
        float(r["choices"][r["gold"]] == r["model_prediction"]["final_answer"])
        for r in records
    ])
    confidences = np.array([r["model_prediction"]["confidence"] for r in records])
    return is_correct, confidences


def compute_all_eces(is_correct: np.ndarray, confidences: np.ndarray) -> dict:
    n     = len(confidences)
    b_opt = optimal_n_bins(n)
    return {
        "ece_uwb_10":  calculate_ece_uwb(is_correct, confidences, n_bins=10),
        "ece_uwb_15":  calculate_ece_uwb(is_correct, confidences, n_bins=15),
        "ece_uwb_opt": calculate_ece_uwb(is_correct, confidences, n_bins=b_opt),
        "ece_umb_10":  calculate_ece_umb(is_correct, confidences, n_bins=10),
        "ece_umb_15":  calculate_ece_umb(is_correct, confidences, n_bins=15),
        "ece_umb_opt": calculate_ece_umb(is_correct, confidences, n_bins=b_opt),
        "n":     n,
        "b_opt": b_opt,
        "accuracy": float(is_correct.mean()),
    }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

# Each dataset config: (results split, calib_dataset, calib_split, instruct_template, base_template)


def build_paths(results_dir: str, dataset: str, instruct_slug: str, base_slug: str) -> dict:
    """Return file paths for all methods. Paths may not exist — caller checks."""
    cfg   = DATASET_CONFIG[dataset]
    split = cfg["apply_split"]
    it    = cfg["instruct_tpl"]
    bt    = cfg["base_tpl"]
    calib = f"calibrated_on_{cfg['calib_dataset']}_{cfg['calib_split']}"
    pp_file = f"{it}_{base_slug}_{bt}.jsonl"

    def _pp(method_dir):
        """Path for any post_processing method that uses the standard pp_file name."""
        return os.path.join(results_dir, dataset, split, "post_processing",
                            method_dir, "confidence", "logprob", instruct_slug, pp_file)

    return {
        "Base":    os.path.join(results_dir, dataset, split, "LLMs", "confidence", "logprob",
                                base_slug, f"{bt}.jsonl"),
        "Instruct": os.path.join(results_dir, dataset, split, "LLMs", "confidence", "logprob",
                                 instruct_slug, f"{it}.jsonl"),
        "Ours":    _pp(f"transformation/confidence/{calib}"),
        "TS":      _pp("temperature_scaling"),
        "HB":      _pp("histogram_binning"),
        "IR":      _pp("isotonic_regression"),
        "MaxConf": _pp("max_confidence"),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

ECE_KEYS   = ["ece_uwb_10", "ece_uwb_15", "ece_uwb_opt",
              "ece_umb_10", "ece_umb_15", "ece_umb_opt"]
ECE_LABELS = ["UWB-10", "UWB-15", "UWB-opt\n(B=⌊n¹ᐟ³⌋)",
              "UMB-10", "UMB-15", "UMB-opt\n(B=⌊n¹ᐟ³⌋)"]

# Method ordering: label-free first, then supervised baselines, then MaxConf aggregation
METHOD_ORDER  = ["Base", "Instruct", "Ours", "TS", "HB", "IR", "MaxConf"]
METHOD_COLORS = {
    "Base":     "#4c72b0",   # blue
    "Instruct": "#dd8452",   # orange
    "Ours":     "#55a868",   # green
    "TS":       "#c44e52",   # red
    "HB":       "#8172b2",   # purple
    "IR":       "#937860",   # brown
    "MaxConf":  "#64b5cd",   # light blue
}

# Which methods are supervised (require labels); used for labeling in output
SUPERVISED_METHODS = {"TS", "HB", "IR"}


def scatter_two_estimators(all_rows, key_x, key_y, label_x, label_y, out_path):
    xs, ys, colors, labels_used = [], [], [], []
    for row in all_rows:
        if key_x not in row or key_y not in row:
            continue
        xs.append(row[key_x])
        ys.append(row[key_y])
        colors.append(METHOD_COLORS.get(row["method"], "grey"))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.8, edgecolors="k", linewidths=0.5, s=60)
    lim = max(max(xs), max(ys)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y = x")
    ax.set_xlabel(label_x, fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")

    # legend patches
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=m) for m, c in METHOD_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.set_title(f"{label_x} vs {label_y}", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def reliability_diagram_row(ax, is_correct, confidences, estimator_name, n_bins,
                             use_umb=False, color="#4c72b0"):
    """Draw one reliability diagram on ax."""
    if is_correct is None:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(estimator_name, fontsize=7)
        return

    n = len(confidences)
    if use_umb:
        order = np.argsort(confidences)
        sc = confidences[order]
        si = is_correct[order]
        splits = np.array_split(np.arange(n), n_bins)
        bin_conf = [sc[idx].mean() for idx in splits if len(idx) > 0]
        bin_acc  = [si[idx].mean() for idx in splits if len(idx) > 0]
        bin_frac = [len(idx) / n   for idx in splits if len(idx) > 0]
    else:
        limits = np.linspace(0.0, 1.0, n_bins + 1)
        bin_conf, bin_acc, bin_frac = [], [], []
        for i, (lo, hi) in enumerate(zip(limits[:-1], limits[1:])):
            mask = confidences <= hi if i == 0 else (confidences > lo) & (confidences <= hi)
            if mask.sum() == 0:
                continue
            bin_conf.append(confidences[mask].mean())
            bin_acc.append(is_correct[mask].mean())
            bin_frac.append(mask.sum() / n)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    for bc, ba, bf in zip(bin_conf, bin_acc, bin_frac):
        ax.bar(bc, ba, width=0.05, color=color, alpha=0.6, edgecolor="none")

    ece_val = sum(abs(bc - ba) * bf for bc, ba, bf in zip(bin_conf, bin_acc, bin_frac))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{estimator_name}\nECE={ece_val:.4f}", fontsize=7)
    ax.set_xlabel("Conf", fontsize=6)
    ax.set_ylabel("Acc",  fontsize=6)
    ax.tick_params(labelsize=6)


def plot_reliability_diagrams(results_dir, dataset, instruct_slug, base_slug, out_path):
    """6-estimator × N-method grid of reliability diagrams for one model pair."""
    paths = build_paths(results_dir, dataset, instruct_slug, base_slug)

    n_methods = len(METHOD_ORDER)
    fig, axes = plt.subplots(n_methods, 6, figsize=(18, n_methods * 3))
    fig.suptitle(f"{dataset} | {instruct_slug} / {base_slug}", fontsize=10)

    estimator_specs = [
        ("UWB-10",  10, False),
        ("UWB-15",  15, False),
        ("UWB-opt", None, False),   # B computed per method
        ("UMB-10",  10, True),
        ("UMB-15",  15, True),
        ("UMB-opt", None, True),
    ]

    for r, method in enumerate(METHOD_ORDER):
        ic, conf = load_confidence_file(paths[method])
        n     = len(conf) if conf is not None else 0
        b_opt = optimal_n_bins(n) if n > 0 else 10
        for c, (name, n_bins, use_umb) in enumerate(estimator_specs):
            ax = axes[r, c]
            nb = b_opt if n_bins is None else n_bins
            label_opt = f" (B={b_opt})" if n_bins is None else ""
            reliability_diagram_row(ax, ic, conf, f"{method}\n{name}{label_opt}",
                                    nb, use_umb=use_umb,
                                    color=METHOD_COLORS[method])
            if r == 0:
                ax.set_title(f"{name}{label_opt}\nECE=?", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(results_dir: str, output_dir: str, datasets=None, models=None):
    os.makedirs(output_dir, exist_ok=True)

    all_rows    = []   # flat list for scatter plots
    json_out    = {}   # nested: dataset -> instruct_slug -> method -> metrics
    rank_check  = {}   # dataset -> instruct_slug -> per-estimator ordering check

    for dataset in (datasets or DATASET_CONFIG):
        json_out[dataset]   = {}
        rank_check[dataset] = {}

        pairs = [(inst, base) for inst, base in discover_pairs(results_dir, dataset)
                 if not models or SLUG_FAMILY[inst] in models]
        if not pairs:
            print(f"[{dataset}] No pairs found, skipping.")
            continue

        print(f"\n[{dataset}] Found {len(pairs)} model pair(s).")

        for instruct_slug, base_slug in pairs:
            pair_key = f"{instruct_slug} / {base_slug}"
            print(f"  Processing: {pair_key}")

            paths  = build_paths(results_dir, dataset, instruct_slug, base_slug)
            entry  = {}
            method_arrays = {}

            for method in METHOD_ORDER:
                ic, conf = load_confidence_file(paths[method])
                if ic is None:
                    print(f"    [{method}] NOT FOUND: {paths[method]}")
                    entry[method] = None
                    method_arrays[method] = (None, None)
                else:
                    metrics = compute_all_eces(ic, conf)
                    entry[method] = metrics
                    method_arrays[method] = (ic, conf)
                    row = {"dataset": dataset, "instruct": instruct_slug,
                           "base": base_slug, "method": method}
                    row.update(metrics)
                    all_rows.append(row)
                    print(f"    [{method:8s}] n={metrics['n']:5d}  "
                          f"b_opt={metrics['b_opt']:3d}  "
                          f"UWB-10={metrics['ece_uwb_10']:.4f}  "
                          f"UWB-opt={metrics['ece_uwb_opt']:.4f}  "
                          f"UMB-10={metrics['ece_umb_10']:.4f}  "
                          f"UMB-opt={metrics['ece_umb_opt']:.4f}")

            json_out[dataset][pair_key] = entry

            # Ranking check: does Base <= Ours < Instruct hold for each estimator?
            # Also records whether Ours beats each supervised baseline.
            checks = {}
            if all(entry[m] is not None for m in ["Base", "Instruct", "Ours"]):
                for key in ECE_KEYS:
                    eb = entry["Base"][key]
                    ei = entry["Instruct"][key]
                    eo = entry["Ours"][key]
                    d = {
                        "base_le_ours":   bool(eb <= eo),
                        "ours_lt_inst":   bool(eo < ei),
                        "base_lt_inst":   bool(eb < ei),
                        "ours_beats_inst": bool(eo < ei),
                    }
                    # Compare Ours vs each supervised baseline when available
                    for sup in ("TS", "HB", "IR"):
                        if entry.get(sup) is not None:
                            d[f"ours_le_{sup.lower()}"] = bool(eo <= entry[sup][key])
                    checks[key] = d
            rank_check[dataset][pair_key] = checks

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    json_path = os.path.join(output_dir, "summary_table.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nSaved: {json_path}")

    # ------------------------------------------------------------------
    # Save ranking check JSON
    # ------------------------------------------------------------------
    rank_path = os.path.join(output_dir, "ranking_check.json")
    with open(rank_path, "w") as f:
        json.dump(rank_check, f, indent=2)
    print(f"Saved: {rank_path}")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if all_rows:
        # Attach category label to each row for CSV
        for row in all_rows:
            m = row["method"]
            if m in SUPERVISED_METHODS:
                row["category"] = "supervised"
            elif m in ("Base", "Instruct", "Ours"):
                row["category"] = "label-free"
            else:
                row["category"] = "aggregation"

        csv_path = os.path.join(output_dir, "summary_table.csv")
        fieldnames = ["dataset", "instruct", "base", "method", "category", "n", "b_opt",
                      "accuracy"] + ECE_KEYS
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved: {csv_path}")

    # ------------------------------------------------------------------
    # Print a compact comparison table to stdout
    # ------------------------------------------------------------------
    HDR_W = 112
    print("\n" + "=" * HDR_W)
    print(f"{'Dataset':<14} {'Instruct':<42} {'Method':<10} {'Category':<12} "
          f"{'UWB10':>7} {'UWB15':>7} {'UWBopt':>7} "
          f"{'UMB10':>7} {'UMB15':>7} {'UMBopt':>7}  b_opt")
    print("=" * HDR_W)
    for row in all_rows:
        m = row["method"]
        if m in SUPERVISED_METHODS:
            cat = "supervised"
        elif m in ("Base", "Instruct", "Ours"):
            cat = "label-free"
        else:
            cat = "aggregation"
        print(f"{row['dataset']:<14} {row['instruct']:<42} {m:<10} {cat:<12} "
              f"{row['ece_uwb_10']:7.4f} {row['ece_uwb_15']:7.4f} {row['ece_uwb_opt']:7.4f} "
              f"{row['ece_umb_10']:7.4f} {row['ece_umb_15']:7.4f} {row['ece_umb_opt']:7.4f}  "
              f"{row['b_opt']}")

    # ------------------------------------------------------------------
    # Scatter plots
    # ------------------------------------------------------------------
    if all_rows:
        scatter_two_estimators(
            all_rows, "ece_uwb_10", "ece_umb_10",
            "UWB-10", "UMB-10",
            os.path.join(output_dir, "scatter_uwb10_vs_umb10.pdf"))

        scatter_two_estimators(
            all_rows, "ece_uwb_10", "ece_uwb_opt",
            "UWB-10", "UWB-opt (B=⌊n^{1/3}⌋)",
            os.path.join(output_dir, "scatter_uwb10_vs_uwb_opt.pdf"))

        scatter_two_estimators(
            all_rows, "ece_umb_10", "ece_umb_opt",
            "UMB-10", "UMB-opt (B=⌊n^{1/3}⌋)",
            os.path.join(output_dir, "scatter_umb10_vs_umb_opt.pdf"))

    # ------------------------------------------------------------------
    # Reliability diagrams for focal model pairs
    # ------------------------------------------------------------------
    focal_pairs = [
        ("MMLU_Redux",    "meta-llama_Llama-3.1-8B-Instruct", "meta-llama_Llama-3.1-8B"),
        ("MMLU_Redux",    "Qwen_Qwen3-8B",                    "Qwen_Qwen3-8B-Base"),
        ("MMLU",          "meta-llama_Llama-3.1-8B-Instruct", "meta-llama_Llama-3.1-8B"),
        ("MMLU",          "Qwen_Qwen3-8B",                    "Qwen_Qwen3-8B-Base"),
        ("CommonSenseQA", "meta-llama_Llama-3.1-8B-Instruct", "meta-llama_Llama-3.1-8B"),
        ("CommonSenseQA", "Qwen_Qwen3-8B",                    "Qwen_Qwen3-8B-Base"),
    ]
    for dataset, instruct_slug, base_slug in focal_pairs:
        if datasets and dataset not in datasets:
            continue
        if models and SLUG_FAMILY[instruct_slug] not in models:
            continue
        slug = f"{dataset}_{instruct_slug.replace('/', '_')}"
        out  = os.path.join(output_dir, f"reliability_{slug}.pdf")
        # check that at least one file exists
        paths = build_paths(results_dir, dataset, instruct_slug, base_slug)
        if any(os.path.exists(p) for p in paths.values()):
            plot_reliability_diagrams(results_dir, dataset, instruct_slug, base_slug, out)
        else:
            print(f"  Skipping reliability diagram for {dataset}/{instruct_slug}: no files found")

    print("\nDone.")


def discover_pairs(results_dir: str, dataset: str) -> list:
    """Discover only supported pairs with an existing calibrated prediction file."""
    return [(inst, base) for inst, base in MODEL_PAIRS
            if os.path.isfile(build_paths(results_dir, dataset, inst, base)["Ours"])]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare fixed-width and equal-mass ECE estimators.")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Root results directory.")
    parser.add_argument("--output_dir",  type=str, default="./results/equal_mass_ece",
                        help="Where to write output files.")
    add_selection_arguments(parser)
    args = parser.parse_args(argv)
    run(args.results_dir, args.output_dir, args.datasets, args.models)


if __name__ == "__main__":
    main()
