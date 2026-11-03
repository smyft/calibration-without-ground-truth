"""Fixed solver-return tests for strictly observational projection diagnostics."""

import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cvxpy as cp
import numpy as np

from source.post_processing.transformation.confidence import solve_minimax_sparse_vectorized


class ProjectionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.args = (np.eye(2) / 2, np.array([0, 1]), np.array([.6, 1.]),
                     np.array([0, 1]), np.array([.6, 1.]), 2)

    @staticmethod
    def fixed_solve(problem, **kwargs):
        for variable in problem.variables():
            if variable.shape == (2, 2):
                variable.save_value(np.array([[300., -2e-7], [200., 500.0000002]]))
            else:
                variable.save_value(np.array([180., 500.]))
        problem._status = cp.OPTIMAL_INACCURATE
        problem._value = -400.
        problem._solver_stats = SimpleNamespace(num_iters=17, setup_time=None, extra_stats=None)

    def run_fixed(self, path=None, implementation=None):
        with patch.object(cp.Problem, "solve", autospec=True,
                          side_effect=implementation or self.fixed_solve) as solve, \
             patch("source.post_processing.transformation.confidence.time.time", side_effect=[10., 12.]), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = solve_minimax_sparse_vectorized(*self.args, diagnostics_path=path)
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(solve.call_args.kwargs, {"solver": cp.CLARABEL, "verbose": False})
        return result

    def test_enabled_diagnostics_preserve_return_values_rng_and_input(self):
        before_arrays = [arg.copy() for arg in self.args[:-1]]
        before_random = random.getstate()
        before_numpy = np.random.get_state()
        disabled = self.run_fixed()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "diagnostics.json")
            enabled = self.run_fixed(path)
            data = json.loads(path.read_text())
        self.assertEqual(enabled, disabled)
        self.assertEqual(enabled[0]["solver_status"], "optimal_inaccurate")
        self.assertEqual(random.getstate(), before_random)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], before_numpy[0])
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        self.assertEqual(after_numpy[2:], before_numpy[2:])
        for before, after in zip(before_arrays, self.args[:-1]):
            np.testing.assert_array_equal(before, after)
        residuals = data["constraint_residuals"]
        for name in ("joint_marginal", "reference_calibration", "reference_mode", "nonnegative_pi"):
            self.assertEqual(residuals[name]["scale_divisor"], 1000.)
        self.assertEqual(residuals["epigraph"]["scale_divisor"], 1000000.)
        scaled = np.array(residuals["epigraph"]["scaled_signed"])
        np.testing.assert_array_equal(residuals["epigraph"]["probability_units_signed"], scaled / 1e6)
        report = data["reports"][1]
        self.assertLess(report["q"]["before_clipping"][0], 0)
        self.assertEqual(report["q"]["after_clipping"][0], 0.)
        self.assertGreater(report["clipping_correction"][0], 0.)
        self.assertNotEqual(data["reports"][0]["raw_map"][1], .6)
        self.assertEqual(data["reports"][0]["deployment_report"], [0, .6])
        self.assertEqual(data["solver_num_iters"], 17)
        self.assertEqual(data["gap"], "unavailable")
        self.assertEqual(data["deployment"], {
            "protocol": "quantized-exact-or-identity-v1", "output_precision": 4,
            "lookup_rule": "exact (choice, quantized_confidence) key; no tolerance",
            "fallback": "retain original choice and quantized confidence for reports retained by the wrapper",
        })

    def test_diagnostic_write_failure_keeps_original_output(self):
        expected = self.run_fixed()
        with tempfile.TemporaryDirectory() as directory:
            invalid_parent = Path(directory, "file")
            invalid_parent.write_text("unchanged")
            observed = self.run_fixed(invalid_parent / "diagnostics.json")
            self.assertEqual(invalid_parent.read_text(), "unchanged")
        self.assertEqual(observed, expected)

    def test_original_solver_error_and_missing_primal_returns_are_preserved(self):
        def missing(problem, **kwargs):
            problem._status = cp.INFEASIBLE
            problem._value = float("inf")

        def solver_error(problem, **kwargs):
            raise cp.error.SolverError("original solver failure")

        for implementation, expected_mapping in ((missing, []), (solver_error, None)):
            with self.subTest(implementation=implementation.__name__), tempfile.TemporaryDirectory() as directory:
                off = self.run_fixed(implementation=implementation)
                path = Path(directory, "diagnostics.json")
                on = self.run_fixed(path, implementation)
                self.assertEqual(on, off)
                self.assertEqual(on[1], expected_mapping)
                self.assertTrue(path.is_file())

    def test_unhandled_solver_exception_is_not_swallowed(self):
        for path in (None, "unused-diagnostics.json"):
            with patch.object(cp.Problem, "solve", side_effect=ValueError("original exception")), \
                 self.assertRaisesRegex(ValueError, "original exception"):
                solve_minimax_sparse_vectorized(*self.args, diagnostics_path=path)


if __name__ == "__main__":
    unittest.main()
