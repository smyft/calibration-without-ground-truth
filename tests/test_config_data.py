"""Model path and dataset contracts, without network access or model loading."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from datasets import Dataset, load_from_disk

from data import data_loader
from src.config import (
    CODE_ROOT, DATASETS, MAIN_PAIRS, MODEL_IDS, MODEL_PAIRS, MODEL_SPECS,
    get_model_spec, resolve_model_path,
)


class ModelConfigTests(unittest.TestCase):
    def test_supported_pairs_and_canonical_result_names(self):
        self.assertEqual(len(MODEL_IDS), 6)
        self.assertEqual(len(MODEL_PAIRS), 3)
        self.assertEqual(len(MAIN_PAIRS), 9)
        self.assertEqual(set(DATASETS), {"MMLU", "MMLU_Redux", "CommonSenseQA"})
        for instruct, base in MODEL_PAIRS:
            inst_spec, base_spec = get_model_spec(instruct), get_model_spec(base)
            self.assertEqual(inst_spec.family, base_spec.family)
            self.assertEqual((inst_spec.variant, base_spec.variant), ("instruct", "base"))
        spec = get_model_spec("mistralai/Ministral-3-8B-Instruct-2512")
        self.assertTrue(spec.local_dir.endswith("-BF16"))
        self.assertFalse(spec.slug.endswith("-BF16"))
        self.assertIs(get_model_spec(spec.local_dir), spec)
        with self.assertRaises(ValueError):
            get_model_spec("unregistered-model")

    def test_complete_checkpoint_and_missing_shard(self):
        spec = next(iter(MODEL_SPECS.values()))
        with tempfile.TemporaryDirectory(dir=CODE_ROOT) as temporary:
            root = Path(temporary)
            checkpoint = root / spec.local_dir
            checkpoint.mkdir()
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                (checkpoint / name).write_text("{}")
            (checkpoint / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"},
            }))
            (checkpoint / "part-1.safetensors").touch()
            with self.assertRaisesRegex(FileNotFoundError, "No download was attempted"):
                resolve_model_path(spec.model_id, model_root=root)
            (checkpoint / "part-2.safetensors").touch()
            self.assertEqual(resolve_model_path(spec.model_id, model_root=root), checkpoint)
            self.assertEqual(resolve_model_path(spec.slug, model_path=checkpoint), checkpoint)
            with patch.dict("os.environ", {"MODEL_ROOT": str(root)}):
                self.assertEqual(resolve_model_path(spec.model_id), checkpoint)


class DatasetContractTests(unittest.TestCase):
    def test_mmlu_ids_subjects_and_labels(self):
        fixture = [{"choices": ["x", "y", "z", "w"], "answer": 2,
                    "question": "Question?", "subject": "anatomy"}]
        with patch.object(data_loader, "load_local_split", return_value=fixture) as load:
            rows = data_loader.prepare_dataset("MMLU", "validation")
        load.assert_called_once_with("MMLU", "all", "validation")
        self.assertEqual(rows, [{"id": 0, "choices": fixture[0]["choices"], "gold": 2,
                                 "task_query": "Question?", "subject": "anatomy"}])

    def test_commonsense_labels_keep_original_option_order(self):
        fixture = [{"id": "sample-id", "answerKey": "D", "question": "Question?",
                    "question_concept": "test", "choices": {
                        "label": ["C", "A", "B", "E", "D"],
                        "text": ["x", "y", "z", "w", "v"],
                    }}]
        with patch.object(data_loader, "load_local_split", return_value=fixture) as load:
            rows = data_loader.prepare_dataset("CommonSenseQA", "train")
        load.assert_called_once_with("CommonSenseQA", "default", "train")
        self.assertEqual(rows[0]["id"], "sample-id")
        self.assertEqual(rows[0]["gold"], 4)
        self.assertEqual(rows[0]["choices"], fixture[0]["choices"]["text"])

    def test_redux_corrections_and_global_ids(self):
        common = {"choices": ["x", "y", "z", "w"], "answer": 1, "question": "Q"}
        fixture = [
            dict(common, error_type="wrong_groundtruth", correct_answer="3"),
            dict(common, error_type="multiple_correct_answers", correct_answer="unknown"),
            dict(common, error_type="ok", correct_answer="0"),
        ]
        with patch.object(data_loader, "load_local_split", return_value=fixture) as load, \
                patch.object(data_loader, "MMLU_SUBJECTS", ("anatomy", "astronomy")):
            rows = data_loader.prepare_dataset("MMLU_Redux")
        self.assertEqual(load.call_args_list, [
            call("MMLU_Redux", "anatomy", "test"),
            call("MMLU_Redux", "astronomy", "test"),
        ])
        self.assertEqual([row["id"] for row in rows], list(range(6)))
        self.assertEqual([row["gold"] for row in rows], [3, 1, 1, 3, 1, 1])
        self.assertEqual([row["subject"] for row in rows], ["anatomy"] * 3 + ["astronomy"] * 3)

    def test_local_records_bypass_prepared_dataset_loading(self):
        record = {"id": 4, "choices": ["x", "y", "z", "w"], "gold": 1,
                  "task_query": "Question?", "subject": "anatomy"}
        with tempfile.TemporaryDirectory(dir=CODE_ROOT) as temporary:
            path = Path(temporary) / "input.jsonl"
            path.write_text(json.dumps(record) + "\n")
            with patch.object(data_loader, "load_local_split") as load:
                self.assertEqual(data_loader.prepare_dataset("MMLU", data_file=path), [record])
            load.assert_not_called()
            path.write_text(json.dumps({"id": 4}) + "\n")
            with self.assertRaisesRegex(ValueError, "missing"):
                data_loader.prepare_dataset("MMLU", data_file=path)

    def test_prepared_datasets_use_environment_root_and_preserve_records(self):
        choices = ["x", "y", "z", "w"]
        mmlu = [
            {"question": "second question", "choices": choices, "answer": 2,
             "subject": "anatomy"},
            {"question": "first question", "choices": choices, "answer": 0,
             "subject": "astronomy"},
        ]
        redux = [
            dict(mmlu[0], error_type="wrong_groundtruth", correct_answer="3"),
            dict(mmlu[1], error_type="multiple_correct_answers", correct_answer="unknown"),
        ]
        csqa = [
            {"id": "second", "question": "second question", "answerKey": "D",
             "question_concept": "test", "choices": {
                 "label": ["C", "A", "B", "E", "D"], "text": choices + ["v"],
             }},
            {"id": "first", "question": "first question", "answerKey": "C",
             "question_concept": "test", "choices": {
                 "label": ["C", "A", "B", "E", "D"], "text": choices + ["v"],
             }},
        ]
        cases = (
            ("MMLU", "MMLU", "all", "validation", mmlu, [0, 1], [2, 0]),
            ("MMLU_Redux", "MMLU-Redux-2.0", "anatomy", "test", redux, [0, 1], [3, 0]),
            ("CommonSenseQA", "CommonSenseQA", "default", "train", csqa,
             ["second", "first"], [4, 0]),
        )
        with tempfile.TemporaryDirectory(dir=CODE_ROOT) as temporary:
            root = Path(temporary)
            for name, local_name, subject, split, fixture, ids, labels in cases:
                with self.subTest(dataset=name):
                    path = root / local_name / "prepared" / subject / split
                    Dataset.from_list(fixture).save_to_disk(str(path))
                    with patch.dict("os.environ", {"DATASET_ROOT": str(root)}), \
                            patch.object(data_loader, "load_from_disk", wraps=load_from_disk) as load, \
                            patch("datasets.load_dataset", side_effect=AssertionError("hub access")):
                        rows = data_loader.prepare_dataset(name, split, subject)
                    load.assert_called_once_with(str(path))
                    self.assertEqual([row["id"] for row in rows], ids)
                    self.assertEqual([row["gold"] for row in rows], labels)
                    self.assertEqual([row["task_query"] for row in rows],
                                     ["second question", "first question"])
                    expected_choices = fixture[0]["choices"]
                    if name == "CommonSenseQA":
                        expected_choices = expected_choices["text"]
                    self.assertEqual([row["choices"] for row in rows], [expected_choices] * 2)

    def test_incomplete_prepared_splits_fail_before_arrow_loading(self):
        cases = (
            ("MMLU", "MMLU", "all", "validation"),
            ("MMLU_Redux", "MMLU-Redux-2.0", "anatomy", "test"),
            ("CommonSenseQA", "CommonSenseQA", "default", "train"),
        )
        with tempfile.TemporaryDirectory(dir=CODE_ROOT) as temporary:
            for name, local_name, subject, split in cases:
                for missing in ("directory", "state.json", "dataset_info.json", "shard"):
                    with self.subTest(dataset=name, missing=missing):
                        root = Path(temporary) / name / missing
                        path = root / local_name / "prepared" / subject / split
                        missing_path = path
                        if missing != "directory":
                            Dataset.from_dict({"question": ["test"]}).save_to_disk(str(path))
                            if missing == "shard":
                                state = json.loads((path / "state.json").read_text())
                                missing_path = path / state["_data_files"][0]["filename"]
                            else:
                                missing_path = path / missing
                            missing_path.unlink()
                        with patch.dict("os.environ", {"DATASET_ROOT": str(root)}), \
                                patch.object(data_loader, "load_from_disk") as load:
                            with self.assertRaises(FileNotFoundError) as error:
                                data_loader.prepare_dataset(name, split, subject)
                        self.assertIn(str(missing_path), str(error.exception))
                        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
