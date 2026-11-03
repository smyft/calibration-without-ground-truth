"""Bregman calibration used by the main experiment.

Fit on aligned calibration reports without labels, then apply the map to the
held-out records. Exact quantized reports use the fitted map; unmatched reports
retain their original answer and quantized confidence. Comparison baselines live in
src.experiments.baselines; their established protocols remain separate.
"""

import copy
import math
import numpy as np

from source.post_processing.transformation.confidence import solve_minimax_sparse_vectorized
from source.post_processing.transformation.report_mapping import (
    output_confidence, select_report_mapping,
)


def _disc(x, bin_width):
    """Round x to the nearest multiple of bin_width (round-half-up)."""
    return round(math.floor(x / bin_width + 0.5) * bin_width, 10)


def build_joint_dist(calib_inst, calib_base, bin_width=0.01):
    """
    Build the joint report distribution P(r1, r2) from calibration records.

    r_i = (class_index, discretized_confidence) for each model.
    Uses align-by-ID matching (robust to ordering differences).

    Returns a dict with keys:
        P_r1r2, r1_j, r1_m, r2_j, r2_m, K,
        records (id/choices/gold only), report_pairs
    Returns None if no aligned pairs are found.
    """
    base_map = {r["id"]: r for r in calib_base}

    if not calib_inst:
        return None

    choices    = calib_inst[0]["choices"]
    K          = len(choices)
    cidx       = {c: i for i, c in enumerate(choices)}

    records_out  = []
    report_pairs = []

    for rec1 in calib_inst:
        rec2 = base_map.get(rec1["id"])
        if rec2 is None:
            continue

        pred1 = rec1["model_prediction"]
        pred2 = rec2["model_prediction"]

        j1 = cidx.get(pred1["final_answer"], -1)
        j2 = cidx.get(pred2["final_answer"], -1)
        if j1 < 0 or j2 < 0:
            continue

        m1 = _disc(float(pred1["confidence"]), bin_width)
        m2 = _disc(float(pred2["confidence"]), bin_width)

        records_out.append({
            "id":      rec1["id"],
            "choices": rec1["choices"],
            "gold":    rec1["gold"],
        })
        report_pairs.append(((j1, m1), (j2, m2)))

    if not report_pairs:
        return None

    unique_r1 = sorted(set(p[0] for p in report_pairs))
    unique_r2 = sorted(set(p[1] for p in report_pairs))
    r1_idx = {r: i for i, r in enumerate(unique_r1)}
    r2_idx = {r: i for i, r in enumerate(unique_r2)}

    P = np.zeros((len(unique_r1), len(unique_r2)))
    for r1, r2 in report_pairs:
        P[r1_idx[r1], r2_idx[r2]] += 1
    P /= len(report_pairs)

    return {
        "P_r1r2":      P,
        "r1_j":        np.array([r[0] for r in unique_r1]),
        "r1_m":        np.array([r[1] for r in unique_r1]),
        "r2_j":        np.array([r[0] for r in unique_r2]),
        "r2_m":        np.array([r[1] for r in unique_r2]),
        "K":           K,
        "records":     records_out,
        "report_pairs": report_pairs,
    }


def fit_bregman(joint_dist, debug=False, diagnostics_path=None):
    """
    Solve the minimax optimization over joint_dist.

    Returns (r1_map, solver_stats):
        r1_map: dict  (j: int, m: float) -> (j_new: int, m_new: float)
        solver_stats: dict with timing and grid info (may be None on failure)
    """
    diagnostic_args = {"diagnostics_path": diagnostics_path} if diagnostics_path is not None else {}
    solver_stats, mapping = solve_minimax_sparse_vectorized(
        joint_dist["P_r1r2"],
        joint_dist["r1_j"],
        joint_dist["r1_m"],
        joint_dist["r2_j"],
        joint_dist["r2_m"],
        joint_dist["K"],
        debug=debug,
        **diagnostic_args,
    )

    if mapping is None:
        return None, solver_stats

    r1_j = joint_dist["r1_j"]
    r1_m = joint_dist["r1_m"]

    r1_map = {}
    for i, (j_new, m_new) in enumerate(mapping):
        key = (int(r1_j[i]), float(r1_m[i]))
        r1_map[key] = (int(j_new), float(np.clip(m_new, 0.0, 1.0)))

    return r1_map, solver_stats


def apply_bregman(r1_map, joint_dist, test_records, bin_width=0.01, precision=4):
    """
    Apply the fitted Bregman mapping to test_records.

    Quantize with the calibration input grid, then look up the complete report.
    On an exact miss, retain the original answer and quantized confidence,
    including absent choices and invalid reports retained by this wrapper.
    ``precision`` is retained for call compatibility; output is always four places.
    """
    r1_j_vals = joint_dist["r1_j"]
    r1_m_vals = joint_dist["r1_m"]

    result = []
    for rec in test_records:
        pred    = rec["model_prediction"]
        choices = rec["choices"]
        cidx    = {c: i for i, c in enumerate(choices)}

        j = cidx.get(pred["final_answer"], -1)
        m = _disc(float(pred["confidence"]), bin_width)

        mapped = select_report_mapping(r1_map, r1_j_vals, r1_m_vals, j, m)

        r = copy.deepcopy(rec)
        if mapped is not None:
            j_new, m_new = mapped
            ans_new = choices[j_new] if j_new < len(choices) else pred["final_answer"]
            r["model_prediction"] = {
                "final_answer": ans_new,
                "confidence":   output_confidence(m_new),
            }
        else:
            r["model_prediction"]["confidence"] = output_confidence(m)
        result.append(r)

    return result
