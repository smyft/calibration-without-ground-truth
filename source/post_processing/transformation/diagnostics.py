"""Read-only observations of the existing confidence projection solve.

Nothing in this module decides whether to accept a solution, retries a solve,
changes an array supplied by the caller, or consumes random numbers. Diagnostic
failures are reported separately and must not replace the original return path.
"""

import json
import math
import sys
from pathlib import Path

import cvxpy
import numpy as np

from .report_mapping import MAPPING_PROTOCOL, OUTPUT_PRECISION, output_confidence


UNAVAILABLE = "unavailable"


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return UNAVAILABLE if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if value is None:
        return UNAVAILABLE
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _public_values(obj):
    result = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            result[name] = UNAVAILABLE
            continue
        if not callable(value):
            result[name] = value
    return result


def _residual(signed, scale, equality=False):
    signed = np.asarray(signed, dtype=float)
    violation = np.abs(signed) if equality else np.maximum(signed, 0.0)
    maximum = float(np.max(violation)) if violation.size else 0.0
    return {
        "kind": "equality" if equality else "inequality_lhs_minus_rhs",
        "scaled_signed": signed.copy(),
        "probability_units_signed": signed / scale,
        "scale_divisor": scale,
        "max_violation_scaled": maximum,
        "max_violation_probability_units": maximum / scale,
    }


class ProjectionDiagnostics:
    """Optional observer; snapshots and persistence are individually guarded."""

    def __init__(self, path):
        self.path = path
        self.q_stages = {}
        self.errors = []

    def _failed(self, error):
        self.errors.append(f"{type(error).__name__}: {error}")
        print(f"Projection diagnostics failed ({self.path}): {error}", file=sys.stderr)

    def _read_backend(self, backend, method):
        try:
            reader = getattr(backend, method, None)
            return _public_values(reader()) if reader is not None else UNAVAILABLE
        except Exception as error:
            self._failed(error)
            return UNAVAILABLE

    def capture_q(self, report_index, stage, values):
        try:
            self.q_stages.setdefault(report_index, {})[stage] = np.array(values, copy=True)
        except Exception as error:
            self._failed(error)

    def finish(self, *, problem, solver_stats, pi, t, u, v, P, P1, P2,
               rows, cols, r1_j, r1_m, r2_j, r2_m, active_a, active_b,
               scale, mapping, debug, error=None):
        try:
            import clarabel

            backend = getattr(problem, "_solver_cache", {}).get("CLARABEL")
            backend_info = backend_settings = UNAVAILABLE
            if backend is not None:
                backend_info = self._read_backend(backend, "get_info")
                backend_settings = self._read_backend(backend, "get_settings")
            stats = problem.solver_stats
            payload = {
                "schema_version": 1,
                "diagnostic_only": True,
                "solver": "CLARABEL",
                "versions": {"cvxpy": cvxpy.__version__, "clarabel": clarabel.__version__},
                "solve_arguments": {"solver": "CLARABEL", "verbose": debug},
                "effective_backend_settings": backend_settings,
                "backend_info": backend_info,
                "status": solver_stats["solver_status"],
                "solver_stats": dict(solver_stats),
                "solver_num_iters": getattr(stats, "num_iters", UNAVAILABLE),
                "solver_setup_time": getattr(stats, "setup_time", UNAVAILABLE),
                "solver_extra_stats": getattr(stats, "extra_stats", UNAVAILABLE),
                "objective_scaled_omitting_constant": problem.value,
                "objective_probability_units_omitting_constant": (
                    problem.value / scale if problem.value is not None else UNAVAILABLE
                ),
                "gap": (({k: backend_info[k] for k in ("gap_abs", "gap_rel") if k in backend_info}
                         or UNAVAILABLE) if isinstance(backend_info, dict) else UNAVAILABLE),
                "scale_factor": scale,
                "error": error,
                "deployment": {"protocol": MAPPING_PROTOCOL, "output_precision": OUTPUT_PRECISION,
                               "lookup_rule": "exact (choice, quantized_confidence) key; no tolerance",
                               "fallback": "retain original choice and quantized confidence for reports retained by the wrapper"},
                "input": {"P_r1r2": np.array(P, copy=True),
                          "r1_j": np.array(r1_j, copy=True), "r1_m": np.array(r1_m, copy=True),
                          "r2_j": np.array(r2_j, copy=True), "r2_m": np.array(r2_m, copy=True)},
                "constraint_residuals": UNAVAILABLE,
                "reports": [],
                "diagnostic_errors": self.errors,
            }
            if pi.value is not None:
                pi_value = np.array(pi.value, copy=True)
                t_value = np.array(t.value, copy=True)
                u_value = np.array(u.value, copy=True)
                v_value = np.array(v.value, copy=True)
                selected_v = v_value[r2_j[active_b], active_b]
                payload["primal"] = {"pi_flat_scaled": pi_value, "pair_rows": rows.copy(),
                                     "pair_cols": cols.copy(), "t_scaled": t_value,
                                     "u_scaled": u_value, "v_scaled": v_value}
                payload["constraint_residuals"] = {
                    "joint_marginal": _residual(pi_value.sum(axis=0) - P[rows, cols] * scale,
                                                scale, equality=True),
                    "reference_calibration": _residual(selected_v - r2_m[active_b] * P2[active_b],
                                                       scale, equality=True),
                    "reference_mode": _residual(v_value[:, active_b] - selected_v[None, :], scale),
                    "nonnegative_pi": _residual(-pi_value, scale),
                    "nonnegative_t": _residual(-t_value, scale),
                    # u and t carry mass scale S; both sides below carry S squared.
                    "epigraph": _residual(u_value[:, active_a] ** 2
                                          - (P1[active_a] * t_value)[None, :], scale ** 2),
                }
            for index, (choice, confidence) in enumerate(zip(r1_j, r1_m)):
                stages = self.q_stages.get(index, {})
                raw, clipped, normalized = (stages.get(stage) for stage in
                                             ("before_clipping", "after_clipping", "after_normalization"))
                entry = {"report": [int(choice), float(confidence)],
                         "report_probability_mass": float(P1[index] / scale),
                         "q": stages or UNAVAILABLE, "raw_map": UNAVAILABLE,
                         "deployment_report": UNAVAILABLE, "mode_gap": UNAVAILABLE}
                if raw is not None and clipped is not None and normalized is not None:
                    entry["q_probability_mass"] = {name: float(values.sum())
                                                   for name, values in stages.items()}
                    entry["clipping_correction"] = clipped - raw
                    entry["normalization_correction"] = normalized - clipped
                    entry["total_correction_l1"] = float(np.abs(normalized - raw).sum())
                    ordered = np.sort(normalized)
                    entry["mode_gap"] = float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else UNAVAILABLE
                    entry["mode_gap_interpretation"] = "numeric difference; no unique-mode guarantee or tie threshold"
                if mapping is not None and index < len(mapping):
                    new_j, new_m = mapping[index]
                    entry["raw_map"] = [int(new_j), float(new_m)]
                    entry["deployment_report"] = [int(new_j), output_confidence(np.clip(new_m, 0.0, 1.0))]
                payload["reports"].append(entry)
            path = Path(self.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                json.dump(_json_value(payload), handle, indent=2, allow_nan=False)
                handle.write("\n")
        except Exception as diagnostic_error:
            self._failed(diagnostic_error)
