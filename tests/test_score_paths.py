"""The score CLI must consume the actual post-processing output layout."""

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from source.post_processing import post_processing as pipeline
from src.experiments.io import _ours_output_path, save_jsonl

scores = importlib.import_module("source.summarize+figures.calculate_scores")
INSTRUCT = "mistralai/Ministral-3-8B-Instruct-2512"
BASE = "mistralai/Ministral-3-8B-Base-2512"


class ScorePathTests(unittest.TestCase):
    def test_supervised_and_max_confidence_outputs_have_one_confidence_directory(self):
        from source.post_processing import max_confidence_aggregation, supervised_baselines

        record = {
            "id": 1, "choices": list("ABCD"), "gold": 0,
            "model_prediction": {"final_answer": "A", "confidence": 1.0},
        }
        for method in ("histogram_binning", "isotonic_regression",
                       "temperature_scaling", "max_confidence"):
            with self.subTest(method=method), tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parents[1]) as directory:
                writer = max_confidence_aggregation if method == "max_confidence" else supervised_baselines
                options = ["--dataset_name", "MMLU_Redux", "--split", "test",
                           "--model1_name", INSTRUCT, "--model2_name", BASE,
                           "--confidence_dir", directory, "--output_dir", directory]
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(writer, "load_data", return_value={
                        ("baseline_thinking", "baseline"): ([record], [record])}))
                    if method != "max_confidence":
                        options += ["--baseline_method", method]
                        stack.enter_context(patch.object(writer, "load_confidence_results",
                                                         return_value=[record]))
                        stack.enter_context(patch.object(writer, "_calibrate_confidences",
                                                         return_value=[1.0]))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    writer.main(options)
                    scores.main(scores.build_parser().parse_args([
                        "--confidence_dir", directory, "--method_type", "post_processing",
                        "--post_processing_method", method, "--llm_model_name", INSTRUCT,
                    ]))
                score_directory = (Path(directory) / "MMLU_Redux" / "test" / "post_processing"
                                   / method / "score" / "logprob" / INSTRUCT.replace("/", "_"))
                score_files = list(score_directory.glob("*.json"))
                self.assertEqual(len(score_files), 1)
                self.assertEqual(json.loads(score_files[0].read_text())["accuracy"], 1.0)

    def test_transformation_defaults_and_transfer_read_the_actual_writer_output(self):
        record = {
            "id": 1, "choices": list("ABCD"), "gold": 0,
            "model_prediction": {"final_answer": "A", "confidence": 1.0},
        }
        for transfer in (False, True):
            with self.subTest(transfer=transfer), tempfile.TemporaryDirectory(
                    dir=Path(__file__).resolve().parents[1]) as directory:
                calibration_dataset = "MMLU" if transfer else "MMLU_Redux"
                calibration_split = "validation" if transfer else "test"
                writer_options = [
                    "--dataset_name", calibration_dataset,
                    "--split", calibration_split,
                    "--model1_name", INSTRUCT, "--model2_name", BASE,
                    "--confidence_dir", directory, "--output_dir", directory,
                ]
                if transfer:
                    writer_options += ["--application_dataset_name", "MMLU_Redux",
                                       "--application_split", "test"]
                # Exercise the real writer without fitting a solver or reading artifacts.
                with patch.object(pipeline, "load_data", return_value={
                        ("baseline_thinking", "baseline"): ([record], [record])}), \
                        patch.object(pipeline, "load_application_data",
                                     return_value={"baseline_thinking": [record]}), \
                        patch.dict(pipeline.TRANSFORMATIONS, {"confidence": (
                            lambda *args, **kwargs: {"records": [record]},
                            lambda *args, **kwargs: ([record], {}),
                        )}), contextlib.redirect_stdout(io.StringIO()):
                    pipeline.post_processing(pipeline.build_parser().parse_args(writer_options))

                prediction_path = Path(_ours_output_path(
                    directory, "MMLU_Redux", "test", calibration_dataset, calibration_split,
                    INSTRUCT.replace("/", "_"), "baseline_thinking",
                    BASE.replace("/", "_"), "baseline"))
                self.assertTrue(prediction_path.is_file())
                stale_path = Path(str(prediction_path).replace(
                    f"/calibrated_on_{calibration_dataset}_{calibration_split}", ""))
                stale_record = {**record, "model_prediction": {"final_answer": "B", "confidence": 1.0}}
                save_jsonl([stale_record], stale_path)

                reader_options = ["--confidence_dir", directory,
                                  "--method_type", "post_processing",
                                  "--llm_model_name", INSTRUCT]
                if transfer:
                    reader_options += ["--calibration_dataset_name", calibration_dataset,
                                       "--calibration_split", calibration_split]
                with contextlib.redirect_stdout(io.StringIO()):
                    scores.main(scores.build_parser().parse_args(reader_options))
                score_path = prediction_path.parents[3] / "score" / "logprob" / prediction_path.parent.name
                score_path = score_path / prediction_path.with_suffix(".json").name
                result = json.loads(score_path.read_text())
                self.assertEqual(result["accuracy"], 1.0)
                self.assertEqual(result["brier_score"], 0.0)

                # Missing current predictions must not silently select stale untagged files.
                prediction_path.unlink()
                score_path.unlink()
                with contextlib.redirect_stdout(io.StringIO()):
                    scores.main(scores.build_parser().parse_args(reader_options))
                self.assertFalse(score_path.exists())
                self.assertEqual(list(Path(directory).rglob("*.json")), [])


if __name__ == "__main__":
    unittest.main()
