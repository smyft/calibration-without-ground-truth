"""
Compute SmoothECE for every method reported in the paper.

The script is deliberately read-only with respect to the saved predictions.  It
loads the 12 paper methods for all 3 datasets and all 3 models, recomputes each
method's correctness from its own prediction file, and writes a new result
bundle without refitting a calibration map or running LLM inference.

Two supports are reported:

* ``native`` uses every row available for that method;
* ``matched`` uses the intersection of sample IDs across all 12 methods in the
  same dataset/model setting.  The matched support is used for paired bootstrap
  confidence intervals and method-to-method differences.

SmoothECE follows ``relplot==1.0.3`` with its automatic reflected-Gaussian
bandwidth.  A source-pinned minimal copy is kept in ``src/third_party`` because
the plotting package is not required by the rest of this project.

Example
-------
    python3 \\
        src/calculating_standard_errors/calculate_smooth_ece.py \\
        --pred-dir results/all_baselines/predictions \\
        --output-dir results/smooth_ece \\
        --bootstrap 1000 \\
        --seed 20260725 \\
        --workers 4
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[2]
SRC_ROOT = CODE_ROOT / "src"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.third_party.relplot_smece import (  # noqa: E402
    RELPlot_COMMIT,
    RELPlot_VERSION,
    SMECE_MESH_POINTS,
    sm_ece,
)


from src.calculating_standard_errors.common import INST_SLUGS, SHORT_NAME


DATASETS = (
    ("MMLU", "MMLU"),
    ("MMLU_Redux", "MMLU-Redux"),
    ("CommonSenseQA", "CommonSenseQA"),
)

MODELS = tuple((slug, SHORT_NAME[slug]) for slug in INST_SLUGS)

# Ordered as in the paper's full comparison tables.
METHODS = (
    ("Base", "Base", "reference"),
    ("Instruct", "Instruct", "uncalibrated"),
    ("TS", "TS", "supervised"),
    ("HB", "HB", "supervised"),
    ("IR", "IR", "supervised"),
    ("GPcalib", "GPcalib", "supervised"),
    ("PBR", "PBR", "supervised"),
    ("BCT", "BCT", "label-free"),
    ("ConfAvg", "ConfAvg", "label-free"),
    ("DACA", "DACA", "label-free"),
    ("Ours_ShuffledRandomRef", "Dummy Reference", "label-free"),
    ("Ours", "Ours", "label-free"),
)

METHOD_ORDER = {display: index for index, (_, display, _) in enumerate(METHODS)}
DATASET_ORDER = {display: index for index, (_, display) in enumerate(DATASETS)}
MODEL_ORDER = {display: index for index, (_, display) in enumerate(MODELS)}

DEFAULT_SEED = 20260725
DEFAULT_BOOTSTRAP = 1000
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)
UPSTREAM_REPOSITORY = "https://github.com/apple/ml-calibration"
UPSTREAM_SOURCE_BLOBS = {
    "src/relplot/metrics.py": "662821b962ea67c21515b1a133a7692ae6ac793d",
    "src/relplot/kernels.py": "a55bec6793ef18ea735e58466e6d5143f5e5b660",
    "src/relplot/config.py": "65059c7f156df78ef44b8bd43dfd85fda216ef8d",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "{}:{} is not valid JSON: {}".format(path, line_number, error)
                )
            if not isinstance(record, dict):
                raise ValueError(
                    "{}:{} must contain a JSON object.".format(path, line_number)
                )
            records.append(record)
    if not records:
        raise ValueError("{} contains no prediction records.".format(path))
    return records


def parse_observations(records, path):
    """Map IDs to each file's own (correctness, confidence, valid-answer flag)."""
    by_id = {}
    for row_number, record in enumerate(records, start=1):
        if "id" not in record:
            raise ValueError("{} row {} has no 'id'.".format(path, row_number))
        sample_id = record["id"]
        try:
            hash(sample_id)
        except TypeError:
            raise ValueError(
                "{} row {} has an unhashable sample id.".format(path, row_number)
            )
        if sample_id in by_id:
            raise ValueError("{} has duplicate id {!r}.".format(path, sample_id))

        choices = record.get("choices")
        gold = record.get("gold")
        prediction = record.get("model_prediction")
        if not isinstance(choices, list) or not choices:
            raise ValueError(
                "{} row {} has invalid choices.".format(path, row_number)
            )
        if (
            not isinstance(gold, int)
            or isinstance(gold, bool)
            or gold < 0
            or gold >= len(choices)
        ):
            raise ValueError("{} row {} has invalid gold.".format(path, row_number))
        if not isinstance(prediction, dict):
            raise ValueError(
                "{} row {} has invalid model_prediction.".format(path, row_number)
            )
        if "final_answer" not in prediction or "confidence" not in prediction:
            raise ValueError(
                "{} row {} has an incomplete model_prediction.".format(
                    path, row_number
                )
            )

        final_answer = prediction["final_answer"]
        try:
            confidence = float(prediction["confidence"])
        except (TypeError, ValueError):
            raise ValueError(
                "{} row {} has non-numeric confidence.".format(path, row_number)
            )
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "{} row {} has confidence outside [0, 1].".format(path, row_number)
            )

        valid_answer = final_answer in choices
        correctness = float(choices[gold] == final_answer)
        by_id[sample_id] = (correctness, confidence, valid_answer)
    return by_id


def sortable_id(sample_id):
    """Stable ordering for integer, string, and other JSON-compatible IDs."""
    return (
        type(sample_id).__name__,
        json.dumps(sample_id, ensure_ascii=False, sort_keys=True),
    )


def arrays_for_ids(observations, sample_ids):
    correctness = np.fromiter(
        (observations[sample_id][0] for sample_id in sample_ids),
        dtype=np.float64,
        count=len(sample_ids),
    )
    confidence = np.fromiter(
        (observations[sample_id][1] for sample_id in sample_ids),
        dtype=np.float64,
        count=len(sample_ids),
    )
    return correctness, confidence


def ece_uwb_10(confidence, correctness):
    edges = np.linspace(0.0, 1.0, 11)
    result = 0.0
    n = len(confidence)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (
            confidence <= upper
            if index == 0
            else ((confidence > lower) & (confidence <= upper))
        )
        count = int(mask.sum())
        if count:
            result += (
                abs(float(confidence[mask].mean() - correctness[mask].mean()))
                * count
                / n
            )
    return float(result)


def load_summary_index(path):
    if path is None or not path.exists():
        return {}
    index = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("dataset"), row.get("model"), row.get("method"))
            if key in index:
                raise ValueError(
                    "Duplicate row in saved code summary for {!r}.".format(key)
                )
            index[key] = row
    return index


def numeric_or_none(value):
    if value is None or value == "":
        return None
    return float(value)


def load_groups(pred_dir, summary_csv):
    summary_index = load_summary_index(summary_csv)
    groups = []
    coverage_rows = []
    point_rows = []

    for dataset_slug, dataset_name in DATASETS:
        for model_slug, model_name in MODELS:
            method_data = {}
            id_sets = []

            for file_stem, display_name, category in METHODS:
                path = pred_dir / dataset_slug / model_slug / (file_stem + ".jsonl")
                if not path.is_file():
                    raise FileNotFoundError(
                        "Required paper-method prediction file is missing: {}".format(
                            path
                        )
                    )
                records = load_jsonl(path)
                observations = parse_observations(records, path)
                method_data[display_name] = {
                    "file_stem": file_stem,
                    "category": category,
                    "path": path,
                    "observations": observations,
                }
                id_sets.append(set(observations))

            common_ids = sorted(set.intersection(*id_sets), key=sortable_id)
            if not common_ids:
                raise ValueError(
                    "No common sample IDs for {}/{}.".format(dataset_name, model_name)
                )
            common_id_hash = sha256_json(common_ids)
            matched_confidence = []
            matched_correctness = []

            for file_stem, display_name, category in METHODS:
                data = method_data[display_name]
                observations = data["observations"]
                native_ids = sorted(observations, key=sortable_id)
                native_correctness, native_confidence = arrays_for_ids(
                    observations, native_ids
                )
                matched_z, matched_c = arrays_for_ids(observations, common_ids)
                matched_correctness.append(matched_z)
                matched_confidence.append(matched_c)

                native_smooth, native_bandwidth = sm_ece(
                    native_confidence, native_correctness, return_width=True
                )
                matched_smooth, matched_bandwidth = sm_ece(
                    matched_c, matched_z, return_width=True
                )

                native_accuracy = float(native_correctness.mean())
                native_ece = ece_uwb_10(native_confidence, native_correctness)
                matched_accuracy = float(matched_z.mean())
                matched_ece = ece_uwb_10(matched_c, matched_z)

                relative_path = str(data["path"].relative_to(CODE_ROOT))
                source_sha256 = sha256_file(data["path"])
                invalid_answer_count = sum(
                    not values[2] for values in observations.values()
                )

                summary_key = (dataset_slug, model_name, file_stem)
                summary = summary_index.get(summary_key)
                summary_n = (
                    int(summary["n"])
                    if summary is not None and summary.get("n")
                    else None
                )
                summary_accuracy = (
                    numeric_or_none(summary.get("acc")) if summary else None
                )
                summary_ece = (
                    numeric_or_none(summary.get("ece")) if summary else None
                )
                summary_n_match = (
                    summary_n == len(native_ids) if summary_n is not None else None
                )
                summary_accuracy_diff = (
                    native_accuracy - summary_accuracy
                    if summary_accuracy is not None
                    else None
                )
                summary_ece_diff = (
                    native_ece - summary_ece if summary_ece is not None else None
                )
                summary_metrics_match = (
                    summary_n_match
                    and abs(summary_accuracy_diff) <= 1e-12
                    and abs(summary_ece_diff) <= 1e-12
                    if summary is not None
                    else None
                )

                coverage_rows.append(
                    {
                        "dataset": dataset_name,
                        "dataset_slug": dataset_slug,
                        "model": model_name,
                        "model_slug": model_slug,
                        "method": display_name,
                        "file_stem": file_stem,
                        "category": category,
                        "source_file": relative_path,
                        "source_sha256": source_sha256,
                        "n_native": len(native_ids),
                        "n_common": len(common_ids),
                        "n_dropped_for_common_support": len(native_ids)
                        - len(common_ids),
                        "common_ids_sha256": common_id_hash,
                        "invalid_answer_count": invalid_answer_count,
                        "confidence_eq_0_native": int(
                            np.count_nonzero(native_confidence == 0.0)
                        ),
                        "confidence_eq_1_native": int(
                            np.count_nonzero(native_confidence == 1.0)
                        ),
                        "confidence_eq_0_matched": int(
                            np.count_nonzero(matched_c == 0.0)
                        ),
                        "confidence_eq_1_matched": int(
                            np.count_nonzero(matched_c == 1.0)
                        ),
                        "saved_summary_row_found": summary is not None,
                        "saved_summary_n": summary_n,
                        "saved_summary_accuracy": summary_accuracy,
                        "saved_summary_ece_uwb_10": summary_ece,
                        "recomputed_accuracy_minus_saved": summary_accuracy_diff,
                        "recomputed_ece_minus_saved": summary_ece_diff,
                        "saved_summary_metrics_match": summary_metrics_match,
                    }
                )

                common_fields = {
                    "dataset": dataset_name,
                    "dataset_slug": dataset_slug,
                    "model": model_name,
                    "model_slug": model_slug,
                    "method": display_name,
                    "file_stem": file_stem,
                    "category": category,
                    "source_file": relative_path,
                    "source_sha256": source_sha256,
                }
                point_rows.append(
                    dict(
                        common_fields,
                        support="native",
                        n=len(native_ids),
                        accuracy=native_accuracy,
                        ece_uwb_10=native_ece,
                        smooth_ece=native_smooth,
                        auto_bandwidth=native_bandwidth,
                    )
                )
                point_rows.append(
                    dict(
                        common_fields,
                        support="matched",
                        n=len(common_ids),
                        accuracy=matched_accuracy,
                        ece_uwb_10=matched_ece,
                        smooth_ece=matched_smooth,
                        auto_bandwidth=matched_bandwidth,
                    )
                )

            groups.append(
                {
                    "dataset": dataset_name,
                    "dataset_slug": dataset_slug,
                    "model": model_name,
                    "model_slug": model_slug,
                    "n": len(common_ids),
                    "common_ids_sha256": common_id_hash,
                    "confidence": np.stack(matched_confidence),
                    "correctness": np.stack(matched_correctness),
                }
            )
            print(
                "[point] {:15s} {:16s} common n={:5d}".format(
                    dataset_name, model_name, len(common_ids)
                ),
                flush=True,
            )

    return groups, coverage_rows, point_rows


def bootstrap_group(group, repetitions, seed, group_index):
    """Run one dataset/model paired test bootstrap in a worker."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, group_index]))
    confidence = group["confidence"]
    correctness = group["correctness"]
    method_count, n = confidence.shape
    values = np.empty((repetitions, method_count), dtype=np.float64)
    bandwidths = np.empty((repetitions, method_count), dtype=np.float64)
    progress_step = max(1, repetitions // 10)

    for repetition in range(repetitions):
        sample = rng.integers(0, n, size=n)
        for method_index in range(method_count):
            value, bandwidth = sm_ece(
                confidence[method_index, sample],
                correctness[method_index, sample],
                return_width=True,
            )
            values[repetition, method_index] = value
            bandwidths[repetition, method_index] = bandwidth
        completed = repetition + 1
        if completed % progress_step == 0 or completed == repetitions:
            print(
                "[bootstrap] {:15s} {:16s} {:4d}/{:4d}".format(
                    group["dataset"], group["model"], completed, repetitions
                ),
                flush=True,
            )

    return {
        "group_index": group_index,
        "dataset": group["dataset"],
        "model": group["model"],
        "values": values,
        "bandwidths": bandwidths,
    }


def run_bootstrap(groups, repetitions, seed, workers):
    if repetitions == 0:
        return {}

    results = {}
    if workers == 1:
        for group_index, group in enumerate(groups):
            result = bootstrap_group(group, repetitions, seed, group_index)
            results[group_index] = result
        return results

    # NumPy performs the convolution-heavy SmoothECE path in compiled code, so
    # threads provide useful group-level parallelism without relying on OS
    # process semaphores (which are unavailable in some sandboxed environments).
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                bootstrap_group, group, repetitions, seed, group_index
            ): group_index
            for group_index, group in enumerate(groups)
        }
        for future in as_completed(futures):
            group_index = futures[future]
            result = future.result()
            results[group_index] = result
            print(
                "[bootstrap] completed {}/{}".format(
                    group_index + 1, len(groups)
                ),
                flush=True,
            )
    return results


def quantile_interval(values, ci_level):
    tail = (1.0 - ci_level) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return float(lower), float(upper)


def make_bootstrap_rows(
    groups, point_rows, bootstrap_results, repetitions, ci_level
):
    matched_points = {
        (row["dataset"], row["model"], row["method"]): row
        for row in point_rows
        if row["support"] == "matched"
    }
    interval_rows = []
    difference_rows = []
    raw_arrays = {}

    for group_index, group in enumerate(groups):
        result = bootstrap_results[group_index]
        values = result["values"]
        bandwidths = result["bandwidths"]
        array_prefix = re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            "{}__{}".format(group["dataset"], group["model"]),
        ).strip("_")
        raw_arrays[array_prefix + "__smooth_ece"] = values
        raw_arrays[array_prefix + "__bandwidth"] = bandwidths

        for method_index, (_, display_name, category) in enumerate(METHODS):
            samples = values[:, method_index]
            width_samples = bandwidths[:, method_index]
            ci_low, ci_high = quantile_interval(samples, ci_level)
            width_low, width_high = quantile_interval(width_samples, ci_level)
            point = matched_points[
                (group["dataset"], group["model"], display_name)
            ]
            interval_rows.append(
                {
                    "dataset": group["dataset"],
                    "model": group["model"],
                    "method": display_name,
                    "category": category,
                    "support": "matched",
                    "n": group["n"],
                    "bootstrap_repetitions": repetitions,
                    "smooth_ece": point["smooth_ece"],
                    "ci_level": ci_level,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_mean": float(samples.mean()),
                    "bootstrap_median": float(np.median(samples)),
                    "bootstrap_std": float(samples.std(ddof=1)),
                    "auto_bandwidth": point["auto_bandwidth"],
                    "bandwidth_ci_low": width_low,
                    "bandwidth_ci_high": width_high,
                    "bandwidth_bootstrap_median": float(np.median(width_samples)),
                }
            )

        ours_index = METHOD_ORDER["Ours"]
        ours_samples = values[:, ours_index]
        ours_point = matched_points[
            (group["dataset"], group["model"], "Ours")
        ]["smooth_ece"]
        for comparator_index, (_, comparator, category) in enumerate(METHODS):
            if comparator == "Ours":
                continue
            deltas = ours_samples - values[:, comparator_index]
            ci_low, ci_high = quantile_interval(deltas, ci_level)
            comparator_point = matched_points[
                (group["dataset"], group["model"], comparator)
            ]["smooth_ece"]
            difference_rows.append(
                {
                    "dataset": group["dataset"],
                    "model": group["model"],
                    "method": "Ours",
                    "comparator": comparator,
                    "comparator_category": category,
                    "support": "matched",
                    "n": group["n"],
                    "bootstrap_repetitions": repetitions,
                    "delta_definition": "Ours - comparator",
                    "delta_smooth_ece": ours_point - comparator_point,
                    "ci_level": ci_level,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_mean": float(deltas.mean()),
                    "bootstrap_median": float(np.median(deltas)),
                    "bootstrap_std": float(deltas.std(ddof=1)),
                    "probability_ours_lower": float(np.mean(deltas < 0.0)),
                    "probability_tie": float(np.mean(deltas == 0.0)),
                }
            )

    return interval_rows, difference_rows, raw_arrays


def normalize_for_json(value):
    if isinstance(value, dict):
        return {key: normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            normalize_for_json(value),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        handle.write("\n")


def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_sort_key(row):
    return (
        DATASET_ORDER[row["dataset"]],
        MODEL_ORDER[row["model"]],
        METHOD_ORDER.get(row.get("method"), len(METHOD_ORDER)),
        row.get("support", ""),
        METHOD_ORDER.get(row.get("comparator"), len(METHOD_ORDER)),
    )


def format_estimate(value, lower, upper):
    return "{:.4f} [{:.4f}, {:.4f}]".format(value, lower, upper)


def render_markdown(interval_rows, difference_rows, coverage_rows, metadata):
    intervals = {
        (row["dataset"], row["model"], row["method"]): row
        for row in interval_rows
    }
    differences = {
        (row["dataset"], row["model"], row["comparator"]): row
        for row in difference_rows
    }
    lines = [
        "# SmoothECE experiment results",
        "",
        "All values use saved per-example predictions from "
        "`results/all_baselines/predictions`; no LLM inference or calibration-map "
        "refit was performed. The main tables use the sample-ID intersection of "
        "all 12 paper methods within each dataset/model setting.",
        "",
        "- Metric: `relplot=={}` automatic-bandwidth SmoothECE "
        "(source-pinned minimal implementation).".format(RELPlot_VERSION),
        "- Bootstrap: {} paired test resamples, seed `{}`, with bandwidth "
        "reselected in every replicate.".format(
            metadata["bootstrap"]["repetitions"], metadata["bootstrap"]["seed"]
        ),
        "- Interval: {:.1f}% percentile interval.".format(
            100 * metadata["bootstrap"]["ci_level"]
        ),
        "- Interpretation: lower SmoothECE is better.",
        "",
    ]

    for _, dataset_name in DATASETS:
        lines.extend(
            [
                "## {}".format(dataset_name),
                "",
                "| Method | Qwen3-8B | LLaMA-3.1-8B | Ministral-3-8B |",
                "|---|---:|---:|---:|",
            ]
        )
        for _, method_name, _ in METHODS:
            cells = []
            for _, model_name in MODELS:
                row = intervals[(dataset_name, model_name, method_name)]
                cells.append(
                    format_estimate(
                        row["smooth_ece"], row["ci_low"], row["ci_high"]
                    )
                )
            lines.append(
                "| {} | {} | {} | {} |".format(
                    method_name, cells[0], cells[1], cells[2]
                )
            )
        sample_sizes = [
            intervals[(dataset_name, model_name, "Ours")]["n"]
            for _, model_name in MODELS
        ]
        lines.extend(
            [
                "",
                "Matched sample sizes (Qwen / LLaMA / Ministral): "
                "`{}` / `{}` / `{}`.".format(*sample_sizes),
                "",
            ]
        )

    lines.extend(
        [
            "## Paired key comparisons",
            "",
            "`Delta = Ours - comparator`; a negative value favors Ours.",
            "",
            "| Dataset | Model | Comparator | Delta | 95% CI | P(Ours lower) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for _, dataset_name in DATASETS:
        for _, model_name in MODELS:
            for comparator in ("Instruct", "DACA", "TS"):
                row = differences[(dataset_name, model_name, comparator)]
                lines.append(
                    "| {} | {} | {} | {:.4f} | [{:.4f}, {:.4f}] | {:.3f} |".format(
                        dataset_name,
                        model_name,
                        comparator,
                        row["delta_smooth_ece"],
                        row["ci_low"],
                        row["ci_high"],
                        row["probability_ours_lower"],
                    )
                )

    instruct_rows = [
        row for row in difference_rows if row["comparator"] == "Instruct"
    ]
    point_improvements = sum(row["delta_smooth_ece"] < 0 for row in instruct_rows)
    significant_improvements = sum(row["ci_high"] < 0 for row in instruct_rows)
    significant_worsenings = sum(row["ci_low"] > 0 for row in instruct_rows)
    max_support_drop = max(
        row["n_dropped_for_common_support"] for row in coverage_rows
    )
    all_summary_match = all(
        row["saved_summary_metrics_match"] is True for row in coverage_rows
    )
    lines.extend(
        [
            "",
            "## Mechanical checks",
            "",
            "- Ours has a lower point-estimate SmoothECE than Instruct in "
            "{} of 9 settings.".format(point_improvements),
            "- The paired 95% interval is wholly below zero in {} settings and "
            "wholly above zero in {} settings.".format(
                significant_improvements, significant_worsenings
            ),
            "- Largest per-method loss from enforcing all-method common support: "
            "{} examples.".format(max_support_drop),
            "- Saved `all_baselines.csv` Acc/ECE/n matches recomputation from "
            "prediction files for all 108 cells: `{}`.".format(all_summary_match),
            "",
            "These intervals condition on the already learned calibration maps. "
            "They measure held-out test-sample uncertainty and do not include "
            "projection-refit uncertainty.",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_output_dir(output_dir):
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(
                "Output path exists and is not a directory: {}".format(output_dir)
            )
        if any(output_dir.iterdir()):
            raise FileExistsError(
                "Refusing to overwrite non-empty output directory: {}".format(
                    output_dir
                )
            )
    else:
        output_dir.mkdir(parents=True)


def run_self_test():
    cases = [
        (
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            0.0,
            0.001953125,
        ),
        (
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.0, 1.0, 1.0, 1.0]),
            0.249993441653965,
            0.25,
        ),
        (
            np.array([0.1, 0.2, 0.35, 0.55, 0.7, 0.9]),
            np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
            0.114948252717964,
            0.115234375,
        ),
    ]
    for confidence, correctness, expected_value, expected_width in cases:
        value, width = sm_ece(confidence, correctness, return_width=True)
        if not np.isclose(value, expected_value, rtol=0.0, atol=1e-12):
            raise AssertionError(
                "SmoothECE self-test value mismatch: {} != {}".format(
                    value, expected_value
                )
            )
        if not np.isclose(width, expected_width, rtol=0.0, atol=1e-15):
            raise AssertionError(
                "SmoothECE self-test bandwidth mismatch: {} != {}".format(
                    width, expected_width
                )
            )

        permutation = np.arange(len(confidence))[::-1]
        permuted = sm_ece(confidence[permutation], correctness[permutation])
        if not np.isclose(permuted, value, rtol=0.0, atol=1e-12):
            raise AssertionError("SmoothECE is not permutation-invariant.")
    print(
        "SmoothECE self-test passed for relplot {} source path.".format(
            RELPlot_VERSION
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-dir",
        "--pred_dir",
        dest="pred_dir",
        type=Path,
        default=CODE_ROOT / "results" / "all_baselines" / "predictions",
        help="Root containing dataset/model/method JSONL prediction files.",
    )
    parser.add_argument(
        "--summary-csv",
        "--summary_csv",
        dest="summary_csv",
        type=Path,
        default=CODE_ROOT / "results" / "all_baselines" / "all_baselines.csv",
        help="Optional saved code summary used only for a read-only audit.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=CODE_ROOT / "results" / "smooth_ece",
        help="New, empty directory for SmoothECE outputs.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help="Number of paired test bootstrap repetitions (0 skips intervals).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel dataset/model bootstrap workers.",
    )
    parser.add_argument(
        "--ci-level",
        "--ci_level",
        dest="ci_level",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run source-equivalence golden tests and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if not 0.0 < args.ci_level < 1.0:
        raise ValueError("--ci-level must lie strictly between 0 and 1.")

    pred_dir = args.pred_dir.resolve()
    summary_csv = args.summary_csv.resolve() if args.summary_csv else None
    output_dir = args.output_dir.resolve()
    if not pred_dir.is_dir():
        raise FileNotFoundError("Prediction directory not found: {}".format(pred_dir))
    prepare_output_dir(output_dir)

    started_at = datetime.now(timezone.utc)
    groups, coverage_rows, point_rows = load_groups(pred_dir, summary_csv)
    if len(groups) != len(DATASETS) * len(MODELS):
        raise AssertionError("Expected exactly 9 dataset/model groups.")
    if len(coverage_rows) != len(DATASETS) * len(MODELS) * len(METHODS):
        raise AssertionError("Expected exactly 108 prediction-file audit rows.")
    if len(point_rows) != 2 * len(coverage_rows):
        raise AssertionError("Expected native and matched point estimates.")

    bootstrap_results = run_bootstrap(
        groups, args.bootstrap, args.seed, min(args.workers, len(groups))
    )
    interval_rows = []
    difference_rows = []
    raw_arrays = {}
    if args.bootstrap:
        interval_rows, difference_rows, raw_arrays = make_bootstrap_rows(
            groups,
            point_rows,
            bootstrap_results,
            args.bootstrap,
            args.ci_level,
        )

    point_rows.sort(key=row_sort_key)
    coverage_rows.sort(key=row_sort_key)
    interval_rows.sort(key=row_sort_key)
    difference_rows.sort(key=row_sort_key)

    finished_at = datetime.now(timezone.utc)
    metadata = {
        "experiment": "SmoothECE on saved paper-method predictions",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
        "llm_inference_run": False,
        "calibration_map_refit": False,
        "prediction_source_of_truth": str(pred_dir),
        "saved_summary_audit_source": str(summary_csv) if summary_csv else None,
        "output_directory": str(output_dir),
        "coverage": {
            "datasets": [display for _, display in DATASETS],
            "models": [display for _, display in MODELS],
            "methods": [display for _, display, _ in METHODS],
            "prediction_cells": len(coverage_rows),
            "point_estimate_rows": len(point_rows),
            "support_scopes": ["native", "matched"],
        },
        "metric": {
            "name": "SmoothECE",
            "upstream_package": "relplot",
            "upstream_version": RELPlot_VERSION,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": RELPlot_COMMIT,
            "upstream_source_blobs": UPSTREAM_SOURCE_BLOBS,
            "implementation": (
                "source-pinned minimal reflected-Gaussian SmoothECE path"
            ),
            "automatic_bandwidth": True,
            "mesh_points_minimum": SMECE_MESH_POINTS,
            "source_file": str(
                SRC_ROOT / "third_party" / "relplot_smece.py"
            ),
            "source_sha256": sha256_file(
                SRC_ROOT / "third_party" / "relplot_smece.py"
            ),
            "license_file": str(
                SRC_ROOT / "third_party" / "APPLE_ML_CALIBRATION_LICENSE.txt"
            ),
        },
        "bootstrap": {
            "repetitions": args.bootstrap,
            "seed": args.seed,
            "workers": min(args.workers, len(groups)),
            "parallel_backend": "threads",
            "ci_level": args.ci_level,
            "interval_type": "percentile",
            "unit": "matched held-out test example",
            "paired_across_methods": True,
            "bandwidth_reselected_each_replicate": True,
            "conditions_on_fixed_learned_maps": True,
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "script": {
            "path": str(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
    }

    point_fields = [
        "dataset",
        "dataset_slug",
        "model",
        "model_slug",
        "method",
        "file_stem",
        "category",
        "support",
        "n",
        "accuracy",
        "ece_uwb_10",
        "smooth_ece",
        "auto_bandwidth",
        "source_file",
        "source_sha256",
    ]
    coverage_fields = [
        "dataset",
        "dataset_slug",
        "model",
        "model_slug",
        "method",
        "file_stem",
        "category",
        "source_file",
        "source_sha256",
        "n_native",
        "n_common",
        "n_dropped_for_common_support",
        "common_ids_sha256",
        "invalid_answer_count",
        "confidence_eq_0_native",
        "confidence_eq_1_native",
        "confidence_eq_0_matched",
        "confidence_eq_1_matched",
        "saved_summary_row_found",
        "saved_summary_n",
        "saved_summary_accuracy",
        "saved_summary_ece_uwb_10",
        "recomputed_accuracy_minus_saved",
        "recomputed_ece_minus_saved",
        "saved_summary_metrics_match",
    ]
    interval_fields = [
        "dataset",
        "model",
        "method",
        "category",
        "support",
        "n",
        "bootstrap_repetitions",
        "smooth_ece",
        "ci_level",
        "ci_low",
        "ci_high",
        "bootstrap_mean",
        "bootstrap_median",
        "bootstrap_std",
        "auto_bandwidth",
        "bandwidth_ci_low",
        "bandwidth_ci_high",
        "bandwidth_bootstrap_median",
    ]
    difference_fields = [
        "dataset",
        "model",
        "method",
        "comparator",
        "comparator_category",
        "support",
        "n",
        "bootstrap_repetitions",
        "delta_definition",
        "delta_smooth_ece",
        "ci_level",
        "ci_low",
        "ci_high",
        "bootstrap_mean",
        "bootstrap_median",
        "bootstrap_std",
        "probability_ours_lower",
        "probability_tie",
    ]

    write_csv(output_dir / "point_estimates.csv", point_rows, point_fields)
    write_json(output_dir / "point_estimates.json", point_rows)
    write_csv(output_dir / "coverage_audit.csv", coverage_rows, coverage_fields)
    write_json(output_dir / "coverage_audit.json", coverage_rows)
    write_json(output_dir / "metadata.json", metadata)

    if args.bootstrap:
        write_csv(
            output_dir / "bootstrap_intervals.csv",
            interval_rows,
            interval_fields,
        )
        write_json(output_dir / "bootstrap_intervals.json", interval_rows)
        write_csv(
            output_dir / "paired_differences.csv",
            difference_rows,
            difference_fields,
        )
        write_json(output_dir / "paired_differences.json", difference_rows)
        np.savez_compressed(
            output_dir / "bootstrap_samples.npz", **raw_arrays
        )
        markdown = render_markdown(
            interval_rows, difference_rows, coverage_rows, metadata
        )
        with open(
            output_dir / "smooth_ece_results.md", "w", encoding="utf-8"
        ) as handle:
            handle.write(markdown)
    print(
        "Wrote {} point estimates and {} bootstrap intervals to {}".format(
            len(point_rows), len(interval_rows), output_dir
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
