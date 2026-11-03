"""Contracts for the label-free main calibration protocol."""

import copy
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from src.calibration import _disc, apply_bregman, build_joint_dist, fit_bregman
from src.experiments.pipeline import apply_calibration_map
from source.post_processing.transformation.report_mapping import select_report_mapping
from source.post_processing.transformation.transformation import transformation_for_confidence
from source.post_processing.utils import calculate_joint_distribution_for_confidence


def record(index, answer, confidence, gold=0):
    return {"id": index, "choices": list("ABCD"), "gold": gold,
            "model_prediction": {"final_answer": answer, "confidence": confidence}}


class CalibrationContractTests(unittest.TestCase):
    def test_joint_distribution_aligns_ids_and_ignores_gold(self):
        instruct = [record(1, "A", 0.51), record(2, "B", 0.72),
                    record(3, "Unknown", 0.25), record(4, "C", 0.9)]
        base = [record(2, "A", 0.6), record(1, "B", 0.4), record(3, "A", 0.5)]
        joint = build_joint_dist(instruct, base)
        np.testing.assert_array_equal(joint["P_r1r2"], [[0, 0.5], [0.5, 0]])
        self.assertEqual([r["id"] for r in joint["records"]], [1, 2])
        changed = copy.deepcopy(instruct)
        for item in changed:
            item["gold"] = 3
        other = build_joint_dist(changed, base)
        for key in ("P_r1r2", "r1_j", "r1_m", "r2_j", "r2_m"):
            np.testing.assert_array_equal(joint[key], other[key])

    def test_exact_mapping_and_quantized_identity_for_missing_reports(self):
        joint = {"r1_j": np.array([0, 0, 1]), "r1_m": np.array([0.3, 0.8, 0.6])}
        mapping = {(0, 0.3): (1, 0.123456), (0, 0.8): (0, 0.75), (1, 0.6): (1, 0.5)}
        inputs = [record(0, "A", 0.3), record(1, "A", 0.77),
                  record(2, "C", 0.8048), record(3, "Unknown", 0.2548)]
        original = copy.deepcopy(inputs)
        outputs = apply_bregman(mapping, joint, inputs)
        self.assertEqual(outputs[0]["model_prediction"], {"final_answer": "B", "confidence": 0.1235})
        self.assertEqual(outputs[1]["model_prediction"], {"final_answer": "A", "confidence": 0.77})
        self.assertEqual(outputs[2]["model_prediction"], {"final_answer": "C", "confidence": 0.8})
        self.assertEqual(outputs[3]["model_prediction"], {"final_answer": "Unknown", "confidence": 0.25})
        self.assertEqual(inputs, original)

    def test_discretization_and_empty_alignment(self):
        self.assertEqual(_disc(0.125, 0.05), 0.15)
        self.assertIsNone(build_joint_dist([], []))
        self.assertIsNone(build_joint_dist([record(1, "A", 0.5)], [record(2, "A", 0.5)]))

    def test_wrappers_share_fixed_map_and_preserve_their_absent_choice_rules(self):
        calibration = {
            "P_r1r2": np.eye(3) / 3, "r1_j": np.array([0, 0, 1]),
            "r1_m": np.array([.25, .75, 1.]), "r2_j": np.array([0, 0, 1]),
            "r2_m": np.array([.25, .75, 1.]), "K": 4,
        }
        values = [(1, .7123456), (0, .8234567), (1, .9999999)]
        mapping = dict(zip(zip(calibration["r1_j"], calibration["r1_m"]), values))
        inputs = [record(9, "A", .25), record(4, "A", .5), record(7, "A", .65),
                  record(1, "A", 0.), record(2, "B", 1.), record(5, "C", .7123456),
                  record(6, "Unknown", .256789)]
        before = copy.deepcopy(inputs)
        application = {
            "records": [{k: r[k] for k in ("id", "choices", "gold")} for r in inputs[:-1]],
            "report_pairs": [((r["choices"].index(r["model_prediction"]["final_answer"]),
                               _disc(r["model_prediction"]["confidence"], .01)), (0, .5))
                             for r in inputs[:-1]],
        }
        main = apply_bregman(mapping, calibration, inputs, precision=2)
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "source.post_processing.transformation.transformation.solve_minimax_sparse_vectorized",
            return_value=({}, values),
        ), patch("src.experiments.pipeline.solve_minimax_sparse_vectorized", return_value=({}, values)):
            canonical, _ = transformation_for_confidence(calibration, precision=2, application_data=application)
            ablation, _ = apply_calibration_map(calibration, application, precision=2)
        self.assertEqual([r["id"] for r in main], [9, 4, 7, 1, 2, 5, 6])
        self.assertEqual([r["id"] for r in canonical], [9, 4, 7, 1, 2])
        self.assertEqual([r["id"] for r in ablation], [9, 4, 7, 1, 2, 5])
        expected = [r["model_prediction"] for r in main[:5]]
        self.assertEqual(expected, [r["model_prediction"] for r in canonical])
        self.assertEqual(expected, [r["model_prediction"] for r in ablation[:5]])
        self.assertEqual(expected, [
            {"final_answer": "B", "confidence": .7123},
            {"final_answer": "A", "confidence": .5},
            {"final_answer": "A", "confidence": .65},
            {"final_answer": "A", "confidence": 0.},
            {"final_answer": "B", "confidence": 1.},
        ])
        self.assertEqual(main[-2]["model_prediction"]["confidence"], .71)
        self.assertEqual(main[-1]["model_prediction"], {"final_answer": "Unknown", "confidence": .26})
        self.assertEqual(ablation[-1]["model_prediction"], {"final_answer": "C", "confidence": .71})
        self.assertEqual(inputs, before)

    def test_plan_example_is_exact_in_all_wrappers(self):
        inputs = [record(0, "A", .7248)]
        before = copy.deepcopy(inputs)
        for exact_hit in (False, True):
            support = [(0, .71), (0, .73), (1, .72)]
            if exact_hit:
                support.append((0, .72))
            support.sort()
            calibration = {
                "P_r1r2": np.eye(len(support)) / len(support),
                "r1_j": np.array([j for j, _ in support]),
                "r1_m": np.array([m for _, m in support]),
                "r2_j": np.array([j for j, _ in support]),
                "r2_m": np.array([m for _, m in support]), "K": 4,
            }
            values = [(2, .8123456)] * len(support)
            mapping = dict(zip(support, values))
            application = {
                "records": [{k: inputs[0][k] for k in ("id", "choices", "gold")}],
                "report_pairs": [((0, .72), (0, .5))],
            }
            expected = ({"final_answer": "C", "confidence": .8123} if exact_hit else
                        {"final_answer": "A", "confidence": .72})
            with self.subTest(exact_hit=exact_hit), contextlib.redirect_stdout(io.StringIO()), patch(
                "source.post_processing.transformation.transformation.solve_minimax_sparse_vectorized",
                return_value=({}, values),
            ), patch("src.experiments.pipeline.solve_minimax_sparse_vectorized", return_value=({}, values)):
                main = apply_bregman(mapping, calibration, inputs)
                canonical, _ = transformation_for_confidence(calibration, application_data=application)
                ablation, _ = apply_calibration_map(calibration, application)
                for output in (main, canonical, ablation):
                    self.assertEqual(output[0]["model_prediction"], expected)
                self.assertEqual(inputs, before)

    def test_helper_never_approximates_an_exact_report_key(self):
        mapping = {(0, .72): (1, .8123456), (1, .73): (2, .9)}
        support_j = np.array([0, 1])
        support_m = np.array([.72, .73])
        self.assertEqual(select_report_mapping(mapping, support_j, support_m, 0, .72),
                         (1, .8123456))
        for choice, confidence in [(0, np.nextafter(.72, 1.)), (0, .73), (-1, .72)]:
            with self.subTest(choice=choice, confidence=confidence):
                self.assertIsNone(select_report_mapping(mapping, support_j, support_m, choice, confidence))

    def test_half_up_boundaries_and_six_grid_misses(self):
        inputs = [record(4, "A", .125), record(3, "A", 0.), record(2, "A", 1.),
                  record(1, "C", .7248), record(0, "Unknown", .7248)]
        before = copy.deepcopy(inputs)
        expected = {
            .1: [.1, 0., 1., .7, .7], .05: [.15, 0., 1., .7, .7],
            .02: [.12, 0., 1., .72, .72], .01: [.13, 0., 1., .72, .72],
            .005: [.125, 0., 1., .725, .725], .002: [.126, 0., 1., .724, .724],
        }
        joint = {"r1_j": np.array([0, 1]), "r1_m": np.array([.4, .72])}
        mapping = {(0, .4): (1, .812345), (1, .72): (1, .912345)}
        for width, confidences in expected.items():
            with self.subTest(width=width):
                outputs = apply_bregman(mapping, joint, inputs, bin_width=width, precision=1)
                self.assertEqual([r["id"] for r in outputs], [4, 3, 2, 1, 0])
                self.assertEqual([r["model_prediction"]["confidence"] for r in outputs], confidences)
                self.assertEqual([r["model_prediction"]["final_answer"] for r in outputs],
                                 ["A", "A", "A", "C", "Unknown"])
                self.assertTrue(all(isinstance(r["model_prediction"]["confidence"], float) for r in outputs))
        self.assertEqual(inputs, before)

    def test_failed_solves_retain_each_wrappers_existing_return_rules(self):
        calibration = build_joint_dist([record(0, "A", .5)], [record(0, "A", .5)])
        for failed_map in (None, []):
            with self.subTest(failed_map=failed_map), contextlib.redirect_stdout(io.StringIO()), patch(
                "src.calibration.solve_minimax_sparse_vectorized", return_value=({}, failed_map),
            ), patch(
                "source.post_processing.transformation.transformation.solve_minimax_sparse_vectorized",
                return_value=({}, failed_map),
            ), patch("src.experiments.pipeline.solve_minimax_sparse_vectorized", return_value=({}, failed_map)):
                main_map, main_stats = fit_bregman(calibration)
                canonical, canonical_stats = transformation_for_confidence(calibration)
                ablation, ablation_stats = apply_calibration_map(calibration, calibration)
                self.assertEqual(main_stats, {})
                self.assertEqual(canonical_stats, {})
                self.assertEqual(ablation_stats, {})
                self.assertIsNone(ablation)
                if failed_map is None:
                    self.assertIsNone(main_map)
                    self.assertIsNone(canonical)
                else:
                    self.assertEqual(main_map, {})
                    self.assertEqual(canonical[0]["model_prediction"],
                                     {"final_answer": "A", "confidence": .5})

    def test_six_input_grids_are_independent_of_output_precision_and_labels(self):
        primary = [record(0, "A", .333), record(1, "A", .817), record(2, "B", .613)]
        reference = [record(0, "B", .647), record(1, "A", .527), record(2, "B", .823)]
        modified = copy.deepcopy(primary)
        for item in modified:
            item["gold"] = 3
            item["model_prediction"]["logprobs_raw"] = {"never_read": "irrelevant"}
        for width in [.1, .05, .02, .01, .005, .002]:
            with self.subTest(width=width), contextlib.redirect_stdout(io.StringIO()):
                joint = calculate_joint_distribution_for_confidence(primary, reference, bin_width=width, precision=2)
                other = calculate_joint_distribution_for_confidence(modified, reference, bin_width=width, precision=9)
                for key in ("P_r1r2", "r1_j", "r1_m", "r2_j", "r2_m"):
                    np.testing.assert_array_equal(joint[key], other[key])
                mapping = {(j, m): (int(j), .71234567) for j, m in zip(joint["r1_j"], joint["r1_m"])}
                result = apply_bregman(mapping, joint, primary, bin_width=width, precision=9)
                self.assertEqual([r["model_prediction"]["confidence"] for r in result], [.7123] * 3)
                changed_result = apply_bregman(mapping, other, modified, bin_width=width, precision=1)
                self.assertEqual([r["model_prediction"] for r in changed_result],
                                 [r["model_prediction"] for r in result])
                missing_inputs = copy.deepcopy(modified)
                for item in missing_inputs:
                    item["model_prediction"]["confidence"] = .4217
                missing = apply_bregman(mapping, joint, missing_inputs, bin_width=width)
                self.assertEqual([r["model_prediction"]["confidence"] for r in missing],
                                 [round(_disc(.4217, width), 4)] * 3)
                missing_joint = calculate_joint_distribution_for_confidence(
                    missing_inputs, reference, bin_width=width, precision=1)
                values = list(mapping.values())
                with patch(
                    "source.post_processing.transformation.transformation.solve_minimax_sparse_vectorized",
                    return_value=({}, values),
                ), patch("src.experiments.pipeline.solve_minimax_sparse_vectorized", return_value=({}, values)):
                    for application, expected in ((joint, result), (missing_joint, missing)):
                        canonical, _ = transformation_for_confidence(joint, application_data=application, precision=1)
                        ablation, _ = apply_calibration_map(joint, application, precision=1)
                        for actual in (canonical, ablation):
                            self.assertEqual([r["model_prediction"]["confidence"] for r in actual],
                                             [r["model_prediction"]["confidence"] for r in expected])


if __name__ == "__main__":
    unittest.main()
