"""Scientific counting and file-safety checks for offline projection tie analysis."""

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.analyze_projection_ties import (
    analyze_diagnostic,
    run_analysis,
    threshold_key,
)


def report(vector, mass, *, raw=None, mode_gap=None):
    entry = {
        "report": [0, 0.6],
        "report_probability_mass": mass,
        "q": "unavailable" if vector is None else {
            "before_clipping": vector if raw is None else raw,
            "after_clipping": vector,
            "after_normalization": vector,
        },
        "raw_map": [0, 0.5],
        "deployment_report": [0, 0.5],
        "mode_gap": "unavailable" if mode_gap is None else mode_gap,
    }
    return entry


def diagnostic(reports, **overrides):
    payload = {
        "schema_version": 1,
        "diagnostic_only": True,
        "status": "optimal",
        "error": "unavailable",
        "diagnostic_errors": [],
        "solver_stats": {"solver_status": "optimal", "n_r1": len(reports)},
        "input": {
            "r1_j": [entry["report"][0] for entry in reports],
            "r1_m": [entry["report"][1] for entry in reports],
        },
        "constraint_residuals": "unavailable",
        "reports": reports,
    }
    payload.update(overrides)
    return payload


class ProjectionTieCountingTests(unittest.TestCase):
    def analyze(self, reports, thresholds=(0.125, 0.25), **overrides):
        return analyze_diagnostic(
            diagnostic(reports, **overrides), Path("diagnostic.json"), thresholds,
        )

    def test_exact_near_boundary_and_mass_denominator_are_distinct(self):
        summary, suspects = self.analyze([
            report([0.5, 0.5], 0.2),
            report([0.5625, 0.4375], 0.3),
            report([0.75, 0.25], 0.4),
            report(None, 0.1),
        ])
        self.assertEqual(summary["valid_bin_count"], 3)
        self.assertEqual(summary["indeterminate_bin_count"], 1)
        self.assertEqual(summary["min_gap"], 0)
        self.assertEqual(summary["exact_tie_count"], 1)
        self.assertAlmostEqual(summary["exact_tie_mass"], 0.2)
        self.assertAlmostEqual(summary["valid_probability_mass"], 0.9)
        self.assertAlmostEqual(summary["indeterminate_probability_mass"], 0.1)
        for tau in (0.125, 0.25):
            key = threshold_key(tau)
            self.assertEqual(summary[f"near_count_{key}"], 1)
            self.assertEqual(summary[f"tie_count_{key}"], 2)
            self.assertAlmostEqual(summary[f"near_mass_{key}"], 0.3)
            self.assertAlmostEqual(summary[f"tie_mass_{key}"], 0.5)
        self.assertEqual({entry["report_index"] for entry in suspects}, {0, 1, 3})

    def test_full_precision_vector_is_not_rounded_before_detection(self):
        summary, suspects = self.analyze(
            [report([0.50004, 0.49996], 1.0)], thresholds=(1e-6, 1e-4),
        )
        self.assertEqual(summary["exact_tie_count"], 0)
        self.assertEqual(summary[f"tie_count_{threshold_key(1e-6)}"], 0)
        self.assertEqual(summary[f"near_count_{threshold_key(1e-4)}"], 1)
        self.assertEqual(suspects[0]["normalized"]["gap"], 0.50004 - 0.49996)
        self.assertEqual(suspects[0]["q"]["after_normalization"], [0.50004, 0.49996])

    def test_vector_and_candidate_sets_are_not_renormalized(self):
        vector = [0.5625, 0.4375, 0.125]
        payload = diagnostic([report(vector, 1.0)])
        before = copy.deepcopy(payload)
        summary, suspects = analyze_diagnostic(payload, Path("diagnostic.json"), (0.125,))
        self.assertEqual(summary[f"near_count_{threshold_key(0.125)}"], 1)
        self.assertEqual(suspects[0]["normalized"]["gap"], 0.125)
        self.assertEqual(suspects[0]["normalized"]["candidates"][threshold_key(0.125)], [0, 1])
        self.assertEqual(payload, before)

    def test_argmax_uses_first_index_and_correction_changes_are_reported(self):
        summary, suspects = self.analyze(
            [report([0.5, 0.5], 1.0, raw=[0.4, 0.6])], thresholds=(0.125,),
        )
        suspect = suspects[0]
        key = threshold_key(0.125)
        self.assertEqual(suspect["normalized"]["argmax"], 0)
        self.assertEqual(suspect["raw"]["argmax"], 1)
        self.assertEqual(suspect["normalized"]["candidates"][key], [0, 1])
        self.assertEqual(suspect["raw"]["candidates"][key], [1])
        self.assertEqual(summary["correction_argmax_changed_count"], 1)
        self.assertEqual(summary["correction_tie_changed_count"], 1)

    def test_invalid_vectors_are_indeterminate_instead_of_separated(self):
        vectors = (
            None, [], [1.0], [[0.5], [0.5]], [True, False],
            ["0.5", "0.5"], ["unavailable", 0.5], ["Infinity", 0.5],
            [float("nan"), 0.5], [float("inf"), 0.5],
        )
        for vector in vectors:
            with self.subTest(vector=vector):
                summary, suspects = self.analyze([report(vector, 1.0)])
                self.assertEqual(summary["valid_bin_count"], 0)
                self.assertEqual(summary["indeterminate_bin_count"], 1)
                self.assertEqual(summary["exact_tie_count"], 0)
                self.assertAlmostEqual(summary["indeterminate_probability_mass"], 1.0)
                self.assertEqual(len(suspects), 1)
                self.assertFalse(suspects[0]["valid"])
                self.assertTrue(suspects[0]["issues"])

    def test_missing_raw_vector_does_not_discard_valid_normalized_vector(self):
        entry = report([0.5, 0.5], 1.0)
        del entry["q"]["before_clipping"]
        summary, suspects = self.analyze([entry])
        self.assertEqual(summary["valid_bin_count"], 1)
        self.assertEqual(summary["exact_tie_count"], 1)
        self.assertEqual(summary["raw_unavailable_bin_count"], 1)
        self.assertTrue(suspects[0]["valid"])

    def test_optimal_inaccurate_is_visible_and_not_automatically_indeterminate(self):
        summary, _ = self.analyze(
            [report([0.75, 0.25], 1.0)], status="optimal_inaccurate",
            solver_stats={"solver_status": "optimal_inaccurate", "n_r1": 1},
        )
        self.assertEqual(summary["solver_status"], "optimal_inaccurate")
        self.assertTrue(summary["optimal_inaccurate"])
        self.assertEqual(summary["valid_bin_count"], 1)
        self.assertEqual(summary["exact_tie_count"], 0)

    def test_real_diagnostic_errors_make_bins_indeterminate(self):
        for overrides in ({"diagnostic_errors": ["snapshot failed"]}, {"error": "solver failed"}):
            with self.subTest(overrides=overrides):
                summary, suspects = self.analyze([report([0.5, 0.5], 1.0)], **overrides)
                self.assertEqual(summary["valid_bin_count"], 0)
                self.assertEqual(summary["indeterminate_bin_count"], 1)
                self.assertEqual(summary["exact_tie_count"], 0)
                self.assertFalse(suspects[0]["valid"])
        for no_error in (None, "unavailable"):
            with self.subTest(no_error=no_error):
                summary, _ = self.analyze([report([0.5, 0.5], 1.0)], error=no_error)
                self.assertEqual(summary["valid_bin_count"], 1)

    def test_missing_probability_mass_is_not_fabricated_or_renormalized(self):
        entries = [report([0.5, 0.5], 0.2), report([0.5, 0.5], "unavailable")]
        summary, _ = self.analyze(entries)
        self.assertEqual(summary["valid_bin_count"], 2)
        self.assertEqual(summary["exact_tie_count"], 2)
        self.assertEqual(summary["missing_mass_count"], 1)
        self.assertFalse(summary["mass_statistics_complete"])
        self.assertAlmostEqual(summary["valid_probability_mass"], 0.2)
        self.assertAlmostEqual(summary["exact_tie_mass"], 0.2)

    def test_stored_mode_gap_is_only_a_cross_check(self):
        summary, _ = self.analyze([report([0.75, 0.25], 1.0, mode_gap=0.0)])
        self.assertEqual(summary["min_gap"], 0.5)
        self.assertEqual(summary["exact_tie_count"], 0)
        self.assertEqual(summary["mode_gap_mismatch_count"], 1)

    def test_unknown_metadata_remains_missing(self):
        summary, _ = self.analyze([report([0.5, 0.5], 1.0)])
        for key in ("dataset", "instruct", "base", "bin_width"):
            self.assertIsNone(summary.get(key), key)

    def test_truncated_vector_is_rejected_when_class_count_is_known(self):
        summary, suspects = self.analyze(
            [report([0.5, 0.5], 1.0)],
            solver_stats={"solver_status": "optimal", "n_r1": 1, "nnz": 10, "n_vars": 40},
        )
        self.assertEqual(summary["valid_bin_count"], 0)
        self.assertEqual(summary["indeterminate_bin_count"], 1)
        self.assertFalse(suspects[0]["valid"])

    def test_missing_report_entries_are_not_dropped_from_expected_count(self):
        summary, suspects = self.analyze(
            [report([0.5, 0.5], 0.2)],
            solver_stats={"solver_status": "optimal", "n_r1": 2},
        )
        self.assertEqual(summary["report_count"], 2)
        self.assertEqual(summary["valid_bin_count"], 1)
        self.assertEqual(summary["indeterminate_bin_count"], 1)
        self.assertEqual(summary["missing_mass_count"], 1)
        self.assertFalse(summary["mass_statistics_complete"])
        self.assertEqual({entry["report_index"] for entry in suspects}, {0, 1})

    def test_grid_width_uses_explicit_filename_and_not_report_confidence(self):
        payload = diagnostic([report([0.5, 0.5], 1.0)])
        summary, _ = analyze_diagnostic(
            payload, Path("diagnostics/MMLU/instruct__base__s-0.02.json"), (0.125,),
        )
        self.assertEqual(summary["dataset"], "MMLU")
        self.assertEqual(summary["instruct"], "instruct")
        self.assertEqual(summary["base"], "base")
        self.assertEqual(summary["bin_width"], 0.02)
        summary, _ = analyze_diagnostic(
            payload, Path("diagnostics/MMLU/instruct__base.json"), (0.125,),
        )
        self.assertIsNone(summary["bin_width"])

    def test_invalid_explicit_width_is_not_replaced_by_documented_default(self):
        payload = diagnostic([report([0.5, 0.5], 1.0)])
        evidence = Path("run_manifest.json")
        for suffix in ("invalid", "nan", "inf", "0", "-0.01"):
            with self.subTest(suffix=suffix):
                summary, _ = analyze_diagnostic(
                    payload, Path(f"diagnostics/MMLU/instruct__base__s-{suffix}.json"),
                    (0.125,), default_bin_width=0.01, metadata_source=evidence,
                )
                self.assertIsNone(summary["bin_width"])
                self.assertIsNone(summary["bin_width_source"])
        summary, _ = analyze_diagnostic(
            payload, Path("diagnostics/MMLU/instruct__base.json"), (0.125,),
            default_bin_width=0.01, metadata_source=evidence,
        )
        self.assertEqual(summary["bin_width"], 0.01)
        self.assertEqual(summary["bin_width_source"], str(evidence))


class ProjectionTieFileSafetyTests(unittest.TestCase):
    def test_analysis_preserves_inputs_and_refuses_any_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "input"
            inputs.mkdir()
            source = inputs / "diagnostic.json"
            source.write_text(json.dumps(diagnostic([report([0.5, 0.5], 1.0)])))
            original_bytes = source.read_bytes()
            output = root / "new-output"
            run_analysis([inputs], output, thresholds=(0.125,))
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual({path.name for path in inputs.iterdir()}, {source.name})
            with (output / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["exact_tie_count"]), 1)
            document = json.loads((output / "suspect_reports.json").read_text())
            self.assertEqual(len(document["suspect_reports"]), 1)
            old_outputs = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaises((FileExistsError, ValueError)):
                run_analysis([inputs], output, thresholds=(0.125,))
            self.assertEqual(old_outputs, {path.name: path.read_bytes() for path in output.iterdir()})
            empty_output = root / "existing-empty"
            empty_output.mkdir()
            with self.assertRaises((FileExistsError, ValueError)):
                run_analysis([inputs], empty_output, thresholds=(0.125,))
            self.assertEqual(list(empty_output.iterdir()), [])

    def test_unreadable_json_is_reported_instead_of_silently_skipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "input"
            inputs.mkdir()
            source = inputs / "broken.json"
            source.write_text('{"reports": [')
            output = root / "output"
            run_analysis([inputs], output)
            with (output / "summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["analysis_errors"])
            self.assertIn(rows[0]["unknown_report_count"].lower(), ("true", "1"))
            self.assertEqual(source.read_text(), '{"reports": [')

    def test_thresholds_reject_nonpositive_or_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "input"
            inputs.mkdir()
            (inputs / "diagnostic.json").write_text(json.dumps(diagnostic([report([0.5, 0.5], 1.0)])))
            for index, invalid in enumerate((0.0, -1.0, float("nan"), float("inf"))):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    run_analysis([inputs], root / f"output-{index}", thresholds=(invalid,))


if __name__ == "__main__":
    unittest.main()
