"""Small numerical and interface checks; no saved experiment files are touched."""

import contextlib
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import numpy as np

from source.post_processing import max_confidence_aggregation as maximum
from source.post_processing import post_processing as pipeline
from source.post_processing import supervised_baselines as baselines
from source.post_processing.records import align_records, parse_record
from source.post_processing.transformation import transformation
from src.calculating_standard_errors import calculate_alternative_ece as alternative
from src.calculating_standard_errors import calculate_standard_errors as standard


def record(uid, confidence, correct=True):
    return {
        "id": uid, "gold": 0, "choices": ["A", "B"],
        "model_prediction": {
            "confidence": confidence, "final_answer": "A" if correct else "B",
        },
    }


def written_records(opened):
    contents = "".join(call.args[0] for call in opened().write.call_args_list)
    return [json.loads(line) for line in contents.splitlines()]


class PostprocessingTests(unittest.TestCase):
    def setUp(self):
        self.primary = [record(1, .2, False), record(2, .5), record(3, .9)]
        self.reference = [record(1, .3), record(2, .5, False), record(3, .8)]
        self.training = [
            record(i, c, i % 3 != 0)
            for i, c in enumerate([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        ]

    def test_alignment_preserves_order_and_duplicate_reference_policy(self):
        reference = [record(3, .6), record(1, .4), record(1, .8)]
        result = align_records(self.primary, reference, include_reference_correctness=True)
        self.assertEqual([r["id"] for r in result], [1, 3])
        self.assertEqual([r["m2_conf"] for r in result], [.8, .6])
        self.assertEqual(parse_record({}), (0.0, "", 0, 0, []))
        self.assertNotIn("m2_correct", baselines.prepare_data(self.primary, reference)[0])
        self.assertIn("m2_correct", maximum.prepare_data(self.primary, reference)[0])

    def test_supervised_baselines_preserve_reference_numerics(self):
        # Frozen outputs from the original implementation, including zero/one inputs.
        expected = {
            "histogram_binning": [1., 1., 0.],
            "isotonic_regression": [.6667, .6667, .6667],
            "temperature_scaling": [.4601, .5, .563],
        }
        for method, confidences in expected.items():
            with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                values = baselines._calibrate_confidences(
                    method,
                    [r["model_prediction"]["confidence"] for r in self.training],
                    [int(r["model_prediction"]["final_answer"] == "A") for r in self.training],
                    [.2, .5, .9],
                )
                self.assertEqual([round(float(c), 4) for c in values], confidences)

    def test_baselines_accept_saved_predictions_without_unused_logits(self):
        args = baselines.build_parser().parse_args(["--baseline_method", "histogram_binning"])
        opened = mock_open()
        pairs = {("baseline_thinking", "baseline"): (self.primary, self.reference)}
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(baselines, "load_data", return_value=pairs), \
             patch.object(baselines, "load_confidence_results", return_value=self.training), \
             patch("os.makedirs"), patch("builtins.open", opened):
            baselines.process_baselines(args)
        rows = written_records(opened)
        self.assertEqual([r["model_prediction"]["confidence"] for r in rows], [1., 1., 0.])
        self.assertEqual([r["model_prediction"]["final_answer"] for r in rows], ["B", "A", "A"])

    def test_maximum_confidence_ties_keep_primary_answer(self):
        args = maximum.build_parser().parse_args([])
        opened = mock_open()
        pairs = {("baseline_thinking", "baseline"): (self.primary, self.reference)}
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(maximum, "load_data", return_value=pairs), \
             patch("os.makedirs"), patch("builtins.open", opened):
            maximum.process_max_confidence_aggregation(args)
        rows = written_records(opened)
        self.assertEqual([r["model_prediction"]["confidence"] for r in rows], [.3, .5, .9])
        self.assertEqual(rows[1]["model_prediction"]["final_answer"], "A")

    def test_binary_transformation_preserves_threshold_and_clipping(self):
        data = {
            "P_f1f2": np.eye(2) / 2, "F1_vals": np.array([.2, .9]),
            "F2_vals": np.array([.3, .8]), "records": self.primary[:2],
            "score_pairs": [(.2, .3), (.9, .8)],
        }
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
            transformation, "solve_calibration_qp", return_value=(None, [-.1, .812345])
        ):
            rows = transformation.transformation_for_binary_prediction(data)
        self.assertEqual(rows[0]["model_prediction"], {"final_answer": "B", "confidence": 1.})
        self.assertEqual(rows[1]["model_prediction"], {"final_answer": "A", "confidence": .8123})

    def test_confidence_application_uses_quantized_identity_and_keeps_choice_filter(self):
        choices = ["A", "B", "C"]
        data = {
            "P_r1r2": np.eye(2) / 2, "r1_j": np.array([0, 1]),
            "r1_m": np.array([.5, .7]), "r2_j": np.array([0, 1]),
            "r2_m": np.array([.6, .8]), "K": 3,
        }
        application = {
            "records": [{"id": i, "choices": choices, "gold": 0} for i in range(3)],
            "report_pairs": [((0, .5), (0, .5)), ((0, .6), (0, .6)), ((2, .8), (2, .8))],
        }
        with contextlib.redirect_stdout(io.StringIO()), patch.object(
            transformation, "solve_minimax_sparse_vectorized",
            return_value=({}, [(1, .812345), (0, .9)]),
        ):
            rows, stats = transformation.transformation_for_confidence(data, application_data=application)
        self.assertEqual([r["id"] for r in rows], [0, 1])
        self.assertEqual(rows[0]["model_prediction"], {"final_answer": "B", "confidence": .8123})
        self.assertEqual(rows[1]["model_prediction"], {"final_answer": "A", "confidence": .6})

    def test_nonconfidence_pipeline_does_not_pass_confidence_bin_width(self):
        for kind in ("mode", "full_prediction"):
            with self.subTest(kind=kind):
                args = pipeline.build_parser().parse_args(["--post_processing_type", kind])
                before = vars(args).copy()
                opened = mock_open()

                def distribution(data1, data2, debug=False, precision=2):
                    return {"records": data1}

                def process(data, debug=False, precision=4, application_data=None):
                    return data["records"]

                with contextlib.redirect_stdout(io.StringIO()), \
                     patch.dict(pipeline.TRANSFORMATIONS, {kind: (distribution, process)}), \
                     patch.object(pipeline, "load_data", return_value={
                         ("baseline_thinking", "baseline"): (self.primary, self.reference)
                     }), patch("os.makedirs"), patch("builtins.open", opened):
                    pipeline.post_processing(args)
                self.assertEqual(written_records(opened), self.primary)
                self.assertEqual(vars(args), before)

    def test_cli_honors_supplied_dataset_model_and_output(self):
        args = pipeline.build_parser().parse_args([
            "--dataset_name", "CommonSenseQA", "--split", "validation",
            "--model1_name", "meta-llama/Llama-3.1-8B-Instruct",
            "--output_dir", "isolated-output", "--precision", "2",
        ])
        self.assertEqual(args.dataset_name, "CommonSenseQA")
        self.assertEqual(args.output_dir, "isolated-output")
        self.assertEqual(args.precision, 2)

    def test_confidence_cli_separates_input_grid_from_fixed_output_precision(self):
        for options, width in (([], .01), (["--precision", "2"], .01),
                               (["--precision", "4"], .0001),
                               (["--precision", "4", "--bin_width", ".05"], .05)):
            with self.subTest(options=options):
                args = pipeline.build_parser().parse_args(options)
                before = vars(args).copy()
                distribution = unittest.mock.Mock(return_value={"records": self.primary})
                process = unittest.mock.Mock(return_value=(self.primary, {}))
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch.dict(pipeline.TRANSFORMATIONS, {"confidence": (distribution, process)}), \
                     patch.object(pipeline, "load_data", return_value={
                         ("baseline_thinking", "baseline"): (self.primary, self.reference)
                     }), patch("os.makedirs"), patch("builtins.open", mock_open()):
                    pipeline.post_processing(args)
                self.assertEqual(distribution.call_args.kwargs["bin_width"], width)
                self.assertEqual(process.call_args.kwargs["precision"], 4)
                self.assertTrue(process.call_args.kwargs["diagnostics_path"].endswith(".jsonl.diagnostics.json"))
                self.assertEqual(vars(args), before)


class StandardErrorTests(unittest.TestCase):
    def test_bin_boundaries_zero_and_one_are_preserved(self):
        confidence = np.array([0., .1, .2, 1.])
        correctness = np.array([0., 1., 0., 1.])
        self.assertEqual(standard.point_ece_uwb(confidence, correctness), .275)
        self.assertEqual(standard.point_ece_uwb(confidence, correctness),
                         alternative.ece_uwb(confidence, correctness, 10))
        self.assertAlmostEqual(standard.se_ece_uwb(confidence, correctness), .225)

    def test_confidence_loss_standard_error_uses_paired_values(self):
        confidence = np.array([.1, .3, .7, .9])
        correctness = np.array([0., 1., 0., 1.])
        losses = (confidence - correctness) ** 2 - correctness
        expected = float(np.std(losses, ddof=1) / 2)
        self.assertEqual(standard.se_cl(confidence, correctness), expected)


if __name__ == "__main__":
    unittest.main()
