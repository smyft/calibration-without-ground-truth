"""Small metadata/description checks; the statistical loops are mocked."""

import contextlib
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments import sample_complexity_reference_validation as reference
from src.experiments.run_all_sample_complexity import PAIRS, select_pairs_with_indices


class SampleComplexityMetadataTests(unittest.TestCase):
    def test_quantile_and_strict_threshold_keep_distinct_criteria(self):
        simple = {
            10: {"ci_lower": 0.0, "separation_rate": 1.0},
            20: {"ci_lower": 0.001, "separation_rate": 0.5},
        }
        self.assertEqual(reference.find_n_star_simple(simple), 20)
        strict = {10: {"separation_rate": 0.79}, 20: {"separation_rate": 0.8}}
        self.assertEqual(reference.find_n_star_strict(strict, 0.8), 20)
        self.assertIsNone(reference.find_n_star_strict(strict, 0.95))

    def test_csqa_selection_preserves_global_pair_indices(self):
        selected = select_pairs_with_indices(SimpleNamespace(datasets=["CommonSenseQA"], models=None))
        self.assertEqual([index for index, _ in selected], [9, 10, 11, 12, 13, 14])
        self.assertTrue(all(PAIRS[index] is config for index, config in selected))

    def test_not_run_metadata_and_repetition_counts_do_not_trigger_bootstrap(self):
        simple = {2: {"separation_rate": 0.9, "mean_gap": 0.2, "ci_lower": 0.0, "ci_upper": 0.5}}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(reference, "load_confidence_file", return_value=[{"id": 0}, {"id": 1}]), \
                patch.object(reference, "extract_arrays", return_value=(np.ones(2), np.zeros(2))), \
                patch.object(reference, "calculate_ece", side_effect=[0.1, 0.2]), \
                patch.object(reference, "run_simple_separation", return_value=simple) as run_simple, \
                patch.object(reference, "run_bootstrap_separation") as run_bootstrap:
            result = reference.run_pair("base.jsonl", "inst.jsonl", "fixture", [2, 10], 7, 11,
                                        0.95, 42, directory, skip_bootstrap=True, threshold_strict=0.8)
        run_bootstrap.assert_not_called()
        self.assertEqual(run_simple.call_args.args[4:], ([2], 7, 42))
        self.assertIsNone(result["n_star_simple"])
        self.assertEqual(result["threshold_strict"], 0.8)
        protocol = result["protocol"]
        self.assertEqual(protocol["simple_status"], "not_reached")
        self.assertEqual(protocol["strict_status"], "not_run")
        self.assertEqual(protocol["outer_subset_repetitions_per_n"], 7)
        self.assertEqual(protocol["bootstrap_repetitions_per_subset"], 0)
        self.assertEqual(protocol["configured_bootstrap_repetitions"], 11)
        self.assertEqual(protocol["simple_seed"], 42)
        self.assertIsNone(protocol["strict_seed"])
        self.assertEqual(protocol["n_grid_effective"], [2])


if __name__ == "__main__":
    unittest.main()
