"""Plan or copy reusable raw inputs into a fresh calibration-revision run.

The default is read-only. ``--copy`` creates a new run and copies only the
explicitly listed raw files. CSQA Instruct outputs (including phase 1), maps,
derived predictions and statistical checkpoints are never copied. This module
does not launch inference, calibration or statistical experiments.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from src.config import CODE_ROOT, MODEL_SPECS


SPLITS = (("MMLU", "validation"), ("MMLU", "test"),
          ("MMLU_Redux", "test"), ("CommonSenseQA", "train"),
          ("CommonSenseQA", "validation"))
EXPECTED_ROWS = {("MMLU", "validation"): 1531, ("MMLU", "test"): 14042,
                 ("MMLU_Redux", "test"): 5700,
                 ("CommonSenseQA", "train"): 9741,
                 ("CommonSenseQA", "validation"): 1221}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_paths():
    """Yield the fixed 33-file reuse allowlist, without discovering extra files."""
    for dataset, split in SPLITS:
        for spec in MODEL_SPECS.values():
            if dataset == "CommonSenseQA" and spec.variant == "instruct":
                continue
            folder = Path(dataset) / split / "LLMs" / "confidence" / "logprob" / spec.slug
            final = "baseline.jsonl" if spec.variant == "base" else "baseline_thinking.jsonl"
            yield dataset, split, spec, folder / final
            if spec.variant == "instruct":
                yield dataset, split, spec, folder / "intermediate" / "baseline_thinking_phase1.jsonl"


def _validate_jsonl(path, expected_rows):
    ids = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row["id"]
            if record_id in ids:
                raise ValueError(f"Duplicate ID {record_id!r} in {path}")
            ids.add(record_id)
            rows += 1
    if rows != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} records, found {rows}")
    return rows


def build_plan(source_predictions, run_dir, *, expected_rows=None):
    """Read and validate original raw files; do not create or modify anything."""
    source = Path(source_predictions).resolve()
    target = Path(run_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Source predictions and destination run must be disjoint trees")
    if source.name == "predictions" and target.is_relative_to(source.parent):
        raise ValueError("The destination must be outside the entire source run")
    if target.exists():
        raise FileExistsError(f"A new, nonexistent run directory is required: {target}")
    counts = EXPECTED_ROWS if expected_rows is None else expected_rows
    files = []
    for dataset, split, spec, relative in raw_paths():
        path = source / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required reusable raw input is missing: {path}")
        # Validate before creating the destination, and preserve the exact bytes.
        rows = _validate_jsonl(path, counts[(dataset, split)])
        files.append({"relative_path": str(relative), "source": str(path),
                      "model_id": spec.model_id, "dataset": dataset, "split": split,
                      "rows": rows, "bytes": path.stat().st_size,
                      "sha256": file_digest(path)})
    rerun = [{"model_id": spec.model_id, "checkpoint_directory_name": spec.local_dir,
              "dataset": "CommonSenseQA", "split": split,
              "records_per_stage": counts[("CommonSenseQA", split)],
              "stages": ["thinking_phase1", "thinking_phase2"]}
             for spec in MODEL_SPECS.values() if spec.variant == "instruct"
             for split in ("train", "validation")]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                            cwd=CODE_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "cvxpy", "clarabel", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {"schema_version": 1, "state": "planned", "source_predictions": str(source),
            "run_dir": str(target), "code_commit": commit, "code_worktree_dirty": dirty,
            "python": sys.version, "packages": versions, "files": files,
            "required_inference": rerun,
            "excluded": ["CommonSenseQA Instruct phase1 and final predictions",
                         "all post-processing, fitted models, maps and score exports",
                         "all sample-complexity checkpoints and result caches"],
            "inference_configuration": "Reuse the original inference manifest's checkpoint/data revisions, prompts, sampling, execution order and seed forwarding; do not add a seed.",
            "reuse_basis": "Raw Base predictions and four-choice Instruct predictions retain their original inference protocol; exact source bytes and SHA256 are recorded.",
            "statistics_reuse": "No statistical files are copied. Unchanged MMLU/Redux statistics may be reused separately only with matching input hashes, protocol, settings and seeds."}


def copy_inputs(plan):
    """Copy a validated allowlist into a new run; never overwrite an old run."""
    target = Path(plan["run_dir"])
    # A stale dry-run plan must not silently accept inputs modified meanwhile.
    for entry in plan["files"]:
        if file_digest(entry["source"]) != entry["sha256"]:
            raise ValueError(f"Source changed after planning: {entry['source']}")
    target.mkdir(parents=True, exist_ok=False)
    manifest = target / "input_manifest.json"
    plan = dict(plan, state="copying", created_at=datetime.now(timezone.utc).isoformat())
    try:
        for entry in plan["files"]:
            destination = target / "predictions" / entry["relative_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Path(entry["source"]).open("rb") as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
            if file_digest(destination) != entry["sha256"]:
                raise ValueError(f"Copy hash mismatch: {destination}")
        plan["state"] = "inputs_prepared_inference_pending"
    except Exception as exc:
        plan.update(state="copy_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # An incomplete batch is preserved and marked; no cleanup or hidden resume.
        manifest.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-predictions", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy reusable raw files into a new run; default only prints a read-only plan.")
    args = parser.parse_args(argv)
    plan = build_plan(args.source_predictions, args.run_dir)
    plan["argv"] = list(sys.argv[1:] if argv is None else argv)
    if args.copy:
        plan = copy_inputs(plan)
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
