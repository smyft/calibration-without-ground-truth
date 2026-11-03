"""Confidence calibration adapters with exact-map or quantized-identity deployment.

Each wrapper retains its original alignment and absent-choice handling.
"""

import types
import numpy as np
from source.post_processing.post_processing import post_processing
from source.post_processing.transformation.confidence import solve_minimax_sparse_vectorized
from source.post_processing.transformation.report_mapping import (
    DEFAULT_BIN_WIDTH, output_confidence, select_report_mapping,
)

PRECISION = 2

def _build_pp_args(results_dir, calib_dataset, calib_split,
                   apply_dataset, apply_split, inst_slug, ref_slug):
    return types.SimpleNamespace(
        dataset_name=calib_dataset,
        split=calib_split,
        calibration_split=calib_split,
        application_dataset_name=apply_dataset,
        application_split=apply_split,
        confidence_dir=results_dir,
        model1_type="LLMs",
        model1_method="logprob",
        model1_name=inst_slug,
        model2_type="LLMs",
        model2_method="logprob",
        model2_name=ref_slug,
        post_processing_method="transformation",
        post_processing_type="confidence",
        precision=PRECISION,
        bin_width=DEFAULT_BIN_WIDTH,
        output_dir=results_dir,
        debug=False,
    )

def run_post_processing(results_dir, calib_dataset, calib_split,
                        apply_dataset, apply_split, inst_slug, ref_slug):
    args = _build_pp_args(results_dir, calib_dataset, calib_split,
                          apply_dataset, apply_split, inst_slug, ref_slug)
    post_processing(args)

def apply_calibration_map(jd_calib, jd_apply, precision=4, diagnostics_path=None):
    """
    Solve the minimax LP on jd_calib, then apply the resulting map to jd_apply.

    Apply the map only to exact quantized report keys. On a miss, retain the
    original choice and quantized confidence, including absent choices.
    ``precision`` is a compatibility argument; output always uses four places.

    Returns (updated_records, solver_stats) or (None, solver_stats) on failure.
    """
    P_r1r2    = jd_calib["P_r1r2"]
    r1_j_vals = jd_calib["r1_j"]
    r1_m_vals = jd_calib["r1_m"]
    r2_j_vals = jd_calib["r2_j"]
    r2_m_vals = jd_calib["r2_m"]
    K         = jd_calib["K"]

    records      = jd_apply["records"]
    report_pairs = jd_apply["report_pairs"]

    # Solve
    diagnostic_args = {"diagnostics_path": diagnostics_path} if diagnostics_path is not None else {}
    solver_stats, r1_map_values = solve_minimax_sparse_vectorized(
        P_r1r2, r1_j_vals, r1_m_vals, r2_j_vals, r2_m_vals, K, **diagnostic_args
    )
    if r1_map_values is None or len(r1_map_values) == 0:
        return None, solver_stats

    # Build lookup: (j, m) -> (new_j, new_m)
    r1_map = {
        (int(r1_j_vals[i]), float(r1_m_vals[i])): r1_map_values[i]
        for i in range(len(r1_map_values))
    }

    updated_records = []
    for record, report_pair in zip(records, report_pairs):
        r1_j = report_pair[0][0]
        r1_m = report_pair[0][1]

        mapped = select_report_mapping(r1_map, r1_j_vals, r1_m_vals, int(r1_j), float(r1_m))
        if mapped is not None:
            new_j, new_m = mapped
            new_m = float(np.clip(new_m, 0.0, 1.0))
            new_m = output_confidence(new_m)
            new_record = record.copy()
            new_record["model_prediction"] = {
                "final_answer": record["choices"][int(new_j)],
                "confidence":   new_m,
            }
        else:
            # Exact support miss: retain the original quantized report.
            new_record = record.copy()
            new_record["model_prediction"] = {
                "final_answer": record["choices"][int(r1_j)],
                "confidence":   output_confidence(r1_m),
            }
        updated_records.append(new_record)

    return updated_records, solver_stats
