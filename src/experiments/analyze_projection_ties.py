"""Inspect saved projection vectors without fitting, inference, or metric updates.

Only q.after_normalization determines primary tie statistics. Stored vectors
are never rounded, clipped, or normalized here. All masses retain their original
calibration-distribution denominator. No numerical threshold certifies a unique
mode of an exact mathematical solution.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DEFAULT_THRESHOLDS = (1e-10, 1e-8, 1e-6, 1e-4)
KNOWN_DATASETS = {"MMLU", "MMLU_Redux", "CommonSenseQA"}


def threshold_key(value):
    return format(value, ".12g")


def validate_thresholds(values):
    values = list(values)
    if not values or any(not _number(t) or t <= 0 for t in values):
        raise ValueError("Thresholds must be finite, strictly positive numbers.")
    values = sorted(set(float(t) for t in values))
    if len({threshold_key(t) for t in values}) != len(values):
        raise ValueError("Threshold labels collide; use more widely separated thresholds.")
    return values


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def vector_statistics(values, thresholds, expected_classes=None):
    """Preserve saved float values and first-index argmax tie breaking."""
    if not isinstance(values, list) or len(values) < 2:
        return None
    if expected_classes is not None and len(values) != expected_classes:
        return None
    if not all(_number(value) for value in values):
        return None
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    first, second = order[:2]
    gap = values[first] - values[second]
    candidates = {
        threshold_key(t): [i for i, value in enumerate(values) if values[first] - value <= t]
        for t in thresholds
    }
    return {
        "argmax": first,
        "top_two_indices": [first, second],
        "top_two_probabilities": [values[first], values[second]],
        "gap": gap,
        "exact_tie": gap == 0,
        "exact_maximizers": [i for i, value in enumerate(values) if value == values[first]],
        "candidates": candidates,
        "classification": {
            threshold_key(t): "exact" if gap == 0 else ("near" if gap <= t else "separated")
            for t in thresholds
        },
    }


def _metadata(path, default_bin_width, metadata_source):
    """Recognize the diagnostic writer's layout; leave other fields missing."""
    path = Path(path)
    metadata = {"dataset": None, "instruct": None, "base": None,
                "bin_width": None, "bin_width_source": None}
    if path.parent.name in KNOWN_DATASETS and path.parent.parent.name == "diagnostics":
        metadata["dataset"] = path.parent.name
        stem = path.stem
        if "__s-" in stem:
            stem, width = stem.rsplit("__s-", 1)
            try:
                width = float(width)
            except ValueError:
                width = None
            if _number(width) and width > 0:
                metadata.update(bin_width=width, bin_width_source="diagnostic_filename")
        parts = stem.split("__")
        if len(parts) == 2 and all(parts):
            metadata.update(instruct=parts[0], base=parts[1])
    if "__s-" not in path.stem and default_bin_width is not None:
        metadata.update(bin_width=default_bin_width, bin_width_source=str(metadata_source))
    metadata["metadata_missing"] = [key for key in ("dataset", "instruct", "base", "bin_width")
                                    if metadata[key] is None]
    return metadata


def _expected_classes(payload):
    stats = payload.get("solver_stats", {})
    if isinstance(stats, dict):
        nnz, variables = stats.get("nnz"), stats.get("n_vars")
        if (_number(nnz) and nnz > 0 and _number(variables) and variables > 0
                and (variables / nnz).is_integer()):
            return int(variables / nnz)
    primal = payload.get("primal", {})
    if isinstance(primal, dict) and isinstance(primal.get("u_scaled"), list):
        return len(primal["u_scaled"])
    return None


def _residual_summary(payload):
    residuals = payload.get("constraint_residuals")
    if not isinstance(residuals, dict):
        return None
    return {
        name: {key: record.get(key) for key in (
            "kind", "scale_divisor", "max_violation_scaled", "max_violation_probability_units")}
        for name, record in residuals.items() if isinstance(record, dict)
    }


def analyze_diagnostic(payload, source_path, thresholds, *, default_bin_width=None,
                       metadata_source=None, source_sha256=None):
    """Return a setting summary and suspect/anomalous reports from one payload."""
    thresholds = validate_thresholds(thresholds)
    source_path = str(source_path)
    file_errors = []
    if not isinstance(payload, dict):
        payload = {}
        file_errors.append("invalid_diagnostic_object")
    status = payload.get("status")
    diagnostic_errors = payload.get("diagnostic_errors", [])
    fatal_errors = []
    if diagnostic_errors:
        fatal_errors.append("diagnostic_generation_errors")
    if payload.get("error") not in (None, "unavailable", ""):
        fatal_errors.append("solver_or_diagnostic_error")
    if status not in ("optimal", "optimal_inaccurate"):
        fatal_errors.append("unavailable_or_unsuccessful_solver_status")
    if payload.get("schema_version") != 1:
        fatal_errors.append("unsupported_or_missing_diagnostic_schema")
    reports = payload.get("reports")
    missing_reports = not isinstance(reports, list)
    if missing_reports:
        reports = []
        file_errors.append("missing_report_list")
    expected_counts = [len(reports)]
    stats = payload.get("solver_stats", {})
    if isinstance(stats, dict) and isinstance(stats.get("n_r1"), int) and not isinstance(stats["n_r1"], bool):
        expected_counts.append(max(0, stats["n_r1"]))
    inputs = payload.get("input", {})
    if isinstance(inputs, dict) and isinstance(inputs.get("r1_j"), list):
        expected_counts.append(len(inputs["r1_j"]))
    count = max(expected_counts)
    if any(n != count for n in expected_counts):
        file_errors.append("report_count_mismatch")
    if not count:
        file_errors.append("no_report_bins")
    file_errors.extend(fatal_errors)
    summary = {
        "source_path": source_path, "source_sha256": source_sha256,
        **_metadata(source_path, default_bin_width, metadata_source),
        "solver_status": status, "optimal_inaccurate": status == "optimal_inaccurate",
        "diagnostic_errors": diagnostic_errors, "analysis_errors": file_errors,
        "unknown_report_count": count == 0,
        "report_count": count if count else None,
        "valid_bin_count": 0, "indeterminate_bin_count": 0,
        "min_gap": None, "exact_tie_count": 0, "exact_tie_mass": 0.0,
        "valid_probability_mass": 0.0, "indeterminate_probability_mass": 0.0,
        "reported_probability_mass": 0.0, "missing_mass_count": 0,
        "mass_statistics_complete": True,
        "raw_unavailable_bin_count": 0, "correction_argmax_changed_count": 0,
        "correction_tie_changed_count": 0, "correction_candidates_changed_count": 0,
        "mode_gap_checked_count": 0, "mode_gap_mismatch_count": 0,
        "constraint_residuals": _residual_summary(payload),
        "residual_units": "linear constraints: probability; epigraph: squared probability; see each scale_divisor",
    }
    for t in thresholds:
        key = threshold_key(t)
        summary.update({f"near_count_{key}": 0, f"tie_count_{key}": 0,
                        f"near_mass_{key}": 0.0, f"tie_mass_{key}": 0.0})
    mass_terms = {key: [] for key in summary
                  if key.endswith("_mass") or key.startswith(("near_mass_", "tie_mass_"))}
    suspects = []
    expected_classes = _expected_classes(payload)
    for index in range(count):
        entry = reports[index] if index < len(reports) else None
        issues = list(fatal_errors)
        if not isinstance(entry, dict):
            entry = {}
            issues.append("missing_or_invalid_report")
        q = entry.get("q")
        stages = q if isinstance(q, dict) else {}
        normalized = vector_statistics(stages.get("after_normalization"), thresholds, expected_classes)
        raw = vector_statistics(stages.get("before_clipping"), thresholds, expected_classes)
        valid = normalized is not None and not fatal_errors
        if normalized is None:
            issues.append("missing_or_invalid_normalized_vector")
        mass = entry.get("report_probability_mass")
        known_mass = _number(mass) and 0 <= mass <= 1
        if not known_mass:
            summary["missing_mass_count"] += 1
            issues.append("missing_or_invalid_probability_mass")
        else:
            mass_terms["reported_probability_mass"].append(mass)
        correction = None
        if raw is None:
            summary["raw_unavailable_bin_count"] += 1
            issues.append("missing_or_invalid_raw_vector")
        elif normalized is not None and len(stages["before_clipping"]) == len(stages["after_normalization"]):
            correction = {
                "argmax_changed": raw["argmax"] != normalized["argmax"],
                "exact_tie_changed": raw["exact_tie"] != normalized["exact_tie"],
                "exact_maximizers_changed": raw["exact_maximizers"] != normalized["exact_maximizers"],
                "classification_changed": {key: raw["classification"][key] != normalized["classification"][key]
                                           for key in normalized["classification"]},
                "candidates_changed": {key: raw["candidates"][key] != normalized["candidates"][key]
                                       for key in normalized["candidates"]},
            }
        elif normalized is not None:
            issues.append("raw_normalized_dimension_mismatch")
            summary["raw_unavailable_bin_count"] += 1
        if valid:
            summary["valid_bin_count"] += 1
            if known_mass:
                mass_terms["valid_probability_mass"].append(mass)
            gap = normalized["gap"]
            summary["min_gap"] = gap if summary["min_gap"] is None else min(summary["min_gap"], gap)
            if gap == 0:
                summary["exact_tie_count"] += 1
                if known_mass:
                    mass_terms["exact_tie_mass"].append(mass)
            for t in thresholds:
                key = threshold_key(t)
                if gap <= t:
                    summary[f"tie_count_{key}"] += 1
                    if known_mass:
                        mass_terms[f"tie_mass_{key}"].append(mass)
                    if gap > 0:
                        summary[f"near_count_{key}"] += 1
                        if known_mass:
                            mass_terms[f"near_mass_{key}"].append(mass)
            saved_gap = entry.get("mode_gap")
            if _number(saved_gap):
                summary["mode_gap_checked_count"] += 1
                if saved_gap != gap:
                    summary["mode_gap_mismatch_count"] += 1
                    issues.append("saved_mode_gap_mismatch")
            if correction is not None:
                summary["correction_argmax_changed_count"] += int(correction["argmax_changed"])
                summary["correction_tie_changed_count"] += int(
                    correction["exact_tie_changed"] or any(correction["classification_changed"].values()))
                summary["correction_candidates_changed_count"] += int(
                    correction["exact_maximizers_changed"] or any(correction["candidates_changed"].values()))
        else:
            summary["indeterminate_bin_count"] += 1
            if known_mass:
                mass_terms["indeterminate_probability_mass"].append(mass)
        changed = correction is not None and (
            correction["argmax_changed"] or correction["exact_tie_changed"]
            or correction["exact_maximizers_changed"] or any(correction["classification_changed"].values())
            or any(correction["candidates_changed"].values()))
        if issues or changed or (normalized is not None and normalized["gap"] <= max(thresholds)):
            suspects.append({
                "source_path": source_path, "solver_status": status,
                "report_index": index, "report": entry.get("report"),
                "report_probability_mass": mass, "valid": valid, "issues": issues,
                "q": q, "normalized": normalized, "raw": raw,
                "correction_comparison": correction,
                "raw_map": entry.get("raw_map"), "deployment_report": entry.get("deployment_report"),
                "saved_mode_gap": entry.get("mode_gap"),
                "clipping_correction": entry.get("clipping_correction"),
                "normalization_correction": entry.get("normalization_correction"),
                "total_correction_l1": entry.get("total_correction_l1"),
            })
    for key, values in mass_terms.items():
        summary[key] = math.fsum(values)
    summary["mass_statistics_complete"] = summary["missing_mass_count"] == 0 and bool(count)
    if not count:
        suspects.append({"source_path": source_path, "solver_status": status,
                         "report_index": None, "valid": False, "issues": file_errors})
    return summary, suspects


def run_analysis(diagnostics_dirs, output_dir, thresholds=DEFAULT_THRESHOLDS, *,
                 default_bin_width=None, metadata_source=None):
    """Read only the explicitly selected directories; reserve a new output root."""
    thresholds = validate_thresholds(thresholds)
    if (default_bin_width is None) != (metadata_source is None):
        raise ValueError("--default-bin-width and --metadata-source must be supplied together.")
    evidence = None
    if default_bin_width is not None:
        if not _number(default_bin_width) or default_bin_width <= 0:
            raise ValueError("The documented bin width must be finite and positive.")
        metadata_source = Path(metadata_source).expanduser().resolve(strict=True)
        evidence = {"path": str(metadata_source),
                    "sha256": hashlib.sha256(metadata_source.read_bytes()).hexdigest(),
                    "default_bin_width": default_bin_width,
                    "scope": "explicit caller-supplied width for files without a filename width"}
    roots = [Path(root).expanduser().resolve(strict=True) for root in diagnostics_dirs]
    paths = set()
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Diagnostic input is not a directory: {root}")
        selected = list(root.rglob("*.json"))
        if not selected:
            raise ValueError(f"No diagnostic JSON files found in: {root}")
        for path in selected:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError(f"Diagnostic symlink leaves its selected directory: {path}")
            paths.add(resolved)
    if not paths:
        raise ValueError("At least one diagnostic input directory is required.")
    output_dir = Path(output_dir).expanduser().resolve()
    if any(output_dir.is_relative_to(root) or root.is_relative_to(output_dir) for root in roots):
        raise ValueError("Output must be separate from every input diagnostic directory.")
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries, suspects, inputs = [], [], []
    for path in sorted(paths):
        source_hash = None
        read_error = None
        try:
            data = path.read_bytes()
            source_hash = hashlib.sha256(data).hexdigest()
            payload = json.loads(data)
        except (OSError, UnicodeError, ValueError) as error:
            payload = None
            read_error = f"{type(error).__name__}: {error}"
        summary, reports = analyze_diagnostic(
            payload, path, thresholds, default_bin_width=default_bin_width,
            metadata_source=metadata_source, source_sha256=source_hash)
        if read_error:
            summary["analysis_errors"].append("unreadable_diagnostic: " + read_error)
        inputs.append({"path": str(path), "sha256": source_hash})
        summaries.append(summary)
        suspects.extend(reports)
    with (output_dir / "summary.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: json.dumps(_json_safe(value), allow_nan=False) if isinstance(value, (dict, list))
                             else value for key, value in row.items()})
    result = {
        "schema_version": 1, "thresholds": thresholds, "inputs": inputs,
        "metadata_evidence": evidence,
        "summary": summaries,
        "interpretation": {
            "primary_vector": "q.after_normalization; no rounding, clipping, or normalization by this analyzer",
            "exact": "gap == 0 in the saved floating-point vector",
            "near": "0 < gap <= threshold", "inclusive_tie": "gap <= threshold",
            "answer_indices": "zero-based; argmax chooses the first maximal coordinate",
            "probability_mass": "sum of original calibration masses; never renormalized; incomplete mass is flagged",
            "failed_diagnostics": "all bins in a file with diagnostic generation errors are conservatively indeterminate",
            "residual_units": "linear constraints use probability units; epigraph uses squared probability units",
            "threshold_limitation": "absolute sensitivity thresholds, not certified coordinate error bounds",
            "scope": "per original diagnostic file; no merging of distinct solves or test-label evaluation",
            "conclusion_limit": "does not prove unique mathematical modes or a theorem's selection condition",
        },
        "suspect_reports": suspects,
    }
    with (output_dir / "suspect_reports.json").open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", nargs="+", required=True,
                        help="Explicit diagnostic directories; recursively read their JSON files only.")
    parser.add_argument("--output-dir", required=True, help="New directory; existing directories are refused.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS,
                        help="Positive absolute probability thresholds.")
    parser.add_argument("--default-bin-width", type=float,
                        help="Documented width for files without __s-<width> in their name; no guessed default.")
    parser.add_argument("--metadata-source", help="Original run file supporting --default-bin-width.")
    args = parser.parse_args(argv)
    result = run_analysis(args.diagnostics_dir, args.output_dir, args.thresholds,
                          default_bin_width=args.default_bin_width, metadata_source=args.metadata_source)
    rows = result["summary"]
    print(f"Analyzed {len(rows)} diagnostic files; "
          f"{sum(row['valid_bin_count'] for row in rows)} valid bins; "
          f"{sum(row['indeterminate_bin_count'] for row in rows)} indeterminate bins; "
          f"{sum(bool(row['analysis_errors']) for row in rows)} files with analysis errors.")
    print(f"Wrote summary.csv and suspect_reports.json to {args.output_dir}")


if __name__ == "__main__":
    main()
