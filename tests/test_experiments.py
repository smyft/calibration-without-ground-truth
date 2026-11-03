"""Regression checks for shared experiment protocols and frozen pre-refactor values."""

import argparse
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.config import MODEL_PAIRS
from src.experiments import baselines, budget_baselines, references
from src.experiments.cli import add_selection_arguments, selected_pairs
from src.experiments.io import _conf_path, load_pair_data, save_jsonl
from src.experiments.pipeline import apply_calibration_map
from src.metrics import compute_metrics


def fixture_records():
    confidences = [0., .1, .2, .35, .51, .7, .9, 1., .45, .8, .61, .22]
    return [
        {
            "id": i if i % 2 == 0 else str(i),
            "choices": list("ABCD"),
            "gold": i % 4,
            "model_prediction": {
                "final_answer": "ABCD"[(i + int(i % 3 == 0)) % 4],
                "confidence": confidence,
            },
            "logprobs_raw": {choice: -float(j + i % 3)
                             for j, choice in enumerate("ABCD")},
        }
        for i, confidence in enumerate(confidences)
    ]


class ExperimentRegressionTests(unittest.TestCase):
    def setUp(self):
        self.records = fixture_records()

    def test_metrics_include_zero_one_and_bin_boundaries(self):
        expected = {
            "accuracy": 2 / 3, "ece": .5216666666666666,
            "brier": .3679666666666667, "cl": -.2986999999999999, "n": 12,
        }
        self.assertEqual(compute_metrics(self.records), expected)

    def test_temperature_is_per_example_binary_nll_and_duplicate_invariant(self):
        from scipy.optimize import minimize_scalar
        from scipy.special import expit
        from src.metrics import is_correct
        confidence = np.clip([r["model_prediction"]["confidence"] for r in self.records],
                             1e-7, 1 - 1e-7)
        logits = np.log(confidence / (1 - confidence))
        labels = np.array([float(is_correct(r)) for r in self.records])
        def objective(temperature):
            p = expit(logits / temperature)
            return -np.mean(labels * np.log(p + 1e-12) +
                            (1 - labels) * np.log(1 - p + 1e-12))
        expected = minimize_scalar(objective, bounds=(.01, 1e5), method="bounded").x
        actual = baselines.fit_temperature_scaling(self.records)
        self.assertAlmostEqual(actual, expected, places=4)
        for repeats in (2, 10):
            duplicated = baselines.fit_temperature_scaling(self.records * repeats)
            self.assertAlmostEqual(duplicated, actual, places=4)
            np.testing.assert_allclose(expit(logits / duplicated), expit(logits / actual), atol=1e-7)
        # Keep the budget implementation's original clipping and numeric protocol.
        with contextlib.redirect_stdout(io.StringIO()):
            _, budget_temperature = budget_baselines.fit_apply_ts(self.records, self.records)
        self.assertAlmostEqual(budget_temperature, 10.52238337676508, places=7)

    def test_temperature_uses_only_confidence_at_deployment(self):
        snapshot = copy.deepcopy(self.records)
        without_raw = [{k: v for k, v in r.items() if k != "logprobs_raw"} for r in self.records]
        temperature = baselines.fit_temperature_scaling(without_raw)
        evaluation = [{k: v for k, v in r.items() if k not in ("gold", "logprobs_raw")}
                      for r in self.records]
        output = baselines.apply_temperature_scaling(temperature, evaluation)
        self.assertEqual(self.records, snapshot)
        for before, after in zip(evaluation, output):
            self.assertEqual(before["id"], after["id"])
            self.assertEqual(before["model_prediction"]["final_answer"],
                             after["model_prediction"]["final_answer"])
            self.assertTrue(np.isfinite(after["model_prediction"]["confidence"]))

    def test_histogram_boundary_predictions(self):
        expected = [1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1.]
        model = baselines.fit_histogram_binning(self.records)
        calibrated = baselines.apply_histogram_binning(model, self.records)
        self.assertEqual([r["model_prediction"]["confidence"] for r in calibrated], expected)
        self.assertEqual(budget_baselines.fit_apply_hb(self.records, self.records), calibrated)

    def test_deterministic_reference_randomness_and_choice_shuffling(self):
        snapshot = copy.deepcopy(self.records)
        random_ref = references.make_random_ref(self.records)
        shuffled = references.make_shuffled_instruct(self.records)
        self.assertEqual([r["model_prediction"]["final_answer"] for r in random_ref],
                         list("ADCBBDAC AACD".replace(" ", "")))
        self.assertEqual([r["gold"] for r in shuffled], [3, 3, 0, 3, 2, 2, 2, 0, 0, 0, 2, 2])
        self.assertEqual(self.records, snapshot)
        for before, after in zip(self.records, shuffled):
            self.assertEqual(
                before["choices"][before["gold"]] == before["model_prediction"]["final_answer"],
                after["choices"][after["gold"]] == after["model_prediction"]["final_answer"],
            )

    def test_full_distribution_missing_data_policies_are_explicit(self):
        record = copy.deepcopy(self.records[0])
        del record["logprobs_raw"]
        self.assertEqual(baselines.get_full_dist(record), dict.fromkeys("ABCD", .25))
        with self.assertRaises(KeyError):
            baselines.get_full_dist(record, allow_uniform=False)

    def test_redux_uses_mmlu_calibration_and_redux_test_files(self):
        instruct, base = MODEL_PAIRS[-1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            for index, (dataset, split, model, template) in enumerate([
                ("MMLU", "validation", instruct, "baseline_thinking"),
                ("MMLU", "validation", base, "baseline"),
                ("MMLU_Redux", "test", instruct, "baseline_thinking"),
                ("MMLU_Redux", "test", base, "baseline"),
            ]):
                save_jsonl([{"id": index}], _conf_path(directory, dataset, split, model, template))
            loaded = load_pair_data(directory, "MMLU_Redux", instruct, base)
            self.assertEqual(loaded, ([{"id": 0}], [{"id": 1}], [{"id": 2}], [{"id": 3}]))

    def test_model_dataset_filters_select_the_requested_pair(self):
        parser = argparse.ArgumentParser()
        add_selection_arguments(parser)
        args = parser.parse_args(["--models", "ministral", "--datasets", "MMLU_Redux"])
        from src.config import MAIN_PAIRS
        self.assertEqual(selected_pairs(MAIN_PAIRS, args), [("MMLU_Redux", *MODEL_PAIRS[-1])])

    def test_dummy_reference_retains_default_datasets(self):
        from src.experiments import experiment_dummy_reference as experiment
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            for options, datasets in [([], ["MMLU_Redux", "CommonSenseQA"]),
                                      (["--datasets", "MMLU"], ["MMLU"])]:
                with patch.object(experiment, "run_pair", return_value=None) as run, \
                        contextlib.redirect_stdout(io.StringIO()):
                    experiment.main(["--output_dir", directory, "--models", "ministral"] + options)
                self.assertEqual([call.args[0] for call in run.call_args_list], datasets)

    def test_filtering_preserves_single_and_multi_seed_assignments(self):
        from src.experiments import run_all_sample_complexity as experiment
        selection = argparse.Namespace(datasets=None, models=["ministral"])
        self.assertEqual([index for index, _ in experiment.select_pairs_with_indices(selection)],
                         [2, 5, 8, 11, 14])
        instruct, base = MODEL_PAIRS[-1]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            for model, template in [(instruct, "baseline_thinking"), (base, "baseline")]:
                save_jsonl(self.records, _conf_path(directory, "MMLU_Redux", "test", model, template))
            for n_seeds, runner, summary, plot, seed_key, expected in [
                (1, "run_pair", "print_summary_table", "plot_combined", "seed", 50),
                (5, "run_pair_multi_seed", "print_multi_seed_summary_table",
                 "plot_combined_multi_seed", "base_seed", 82),
            ]:
                with patch.object(experiment, runner, return_value={}) as run, \
                        patch.object(experiment, summary), patch.object(experiment, plot), \
                        contextlib.redirect_stdout(io.StringIO()):
                    experiment.main([
                        "--models", "ministral", "--datasets", "MMLU_Redux",
                        "--results_dir", directory,
                        "--output_dir", str(Path(directory, "output", str(n_seeds))),
                        "--n_seeds", str(n_seeds),
                    ])
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.kwargs[seed_key], expected)

    def test_precision_ablation_uses_quantized_identity_for_support_misses(self):
        calibration = {
            "P_r1r2": np.array([[1.]]), "r1_j": np.array([0]),
            "r1_m": np.array([.5]), "r2_j": np.array([0]),
            "r2_m": np.array([.5]), "K": 4,
        }
        application = {
            "records": [{"id": i, "choices": list("ABCD"), "gold": 0} for i in range(3)],
            "report_pairs": [((0, .5), (0, .5)), ((0, .6), (0, .5)), ((2, .7), (0, .5))],
        }
        with patch("src.experiments.pipeline.solve_minimax_sparse_vectorized",
                   return_value=({"solver_status": "optimal"}, [(1, .765432)])):
            records, _ = apply_calibration_map(calibration, application)
        self.assertEqual([r["model_prediction"] for r in records], [
            {"final_answer": "B", "confidence": .7654},
            {"final_answer": "A", "confidence": .6},
            {"final_answer": "C", "confidence": .7},
        ])


if __name__ == "__main__":
    unittest.main()
