import json
from pathlib import Path
import tempfile
import unittest

from src.experiments.revision_inputs import SPLITS, build_plan, copy_inputs, file_digest, raw_paths


class RevisionInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "old" / "predictions"
        self.target = self.root / "new"
        self.counts = {pair: 2 for pair in SPLITS}
        for _, _, _, relative in raw_paths():
            p = self.source / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(''.join(json.dumps({"id": i, "payload": "unchanged"}) + '\n' for i in range(2)))
        # Deliberate traps: glob-copying would incorrectly import these artifacts.
        for relative in ("CommonSenseQA/train/LLMs/confidence/logprob/Qwen_Qwen3-8B/baseline_thinking.jsonl",
                         "CommonSenseQA/train/LLMs/confidence/logprob/Qwen_Qwen3-8B/intermediate/baseline_thinking_phase1.jsonl",
                         "MMLU/test/post_processing/stale_map.json",
                         "sample_complexity/multi_seed_results.json"):
            p = self.source / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('{"must_not_copy": true}\n')

    def plan(self):
        return build_plan(self.source, self.target, expected_rows=self.counts)

    def test_plan_is_read_only_and_copy_excludes_stale_inputs(self):
        before = {str(p): file_digest(p) for p in self.source.rglob('*') if p.is_file()}
        plan = self.plan()
        self.assertFalse(self.target.exists())
        self.assertEqual(len(plan["files"]), 33)
        self.assertEqual(len(plan["required_inference"]), 6)
        self.assertEqual(sum(len(r["stages"]) for r in plan["required_inference"]), 12)
        copy_inputs(plan)
        copied = sorted(str(p.relative_to(self.target / "predictions")) for p in (self.target / "predictions").rglob('*') if p.is_file())
        self.assertEqual(copied, sorted(r["relative_path"] for r in plan["files"]))
        self.assertEqual(before, {str(p): file_digest(p) for p in self.source.rglob('*') if p.is_file()})
        self.assertEqual(json.loads((self.target / "input_manifest.json").read_text())["state"], "inputs_prepared_inference_pending")
        with self.assertRaises(FileExistsError):
            copy_inputs(plan)

    def test_missing_duplicate_or_incomplete_input_rejected_before_writes(self):
        p = self.source / next(raw_paths())[3]
        saved = p.read_bytes()
        p.unlink()
        with self.assertRaises(FileNotFoundError):
            self.plan()
        p.write_text('{"id": 0}\n{"id": 0}\n')
        with self.assertRaisesRegex(ValueError, "Duplicate ID"):
            self.plan()
        p.write_text('{"id": 0}\n')
        with self.assertRaisesRegex(ValueError, "expected 2 records"):
            self.plan()
        p.write_bytes(saved)
        self.assertFalse(self.target.exists())

    def test_overlap_existing_target_and_stale_plan_rejected(self):
        with self.assertRaises(ValueError):
            build_plan(self.source, self.source / "new", expected_rows=self.counts)
        with self.assertRaises(ValueError):
            build_plan(self.source, self.source.parent, expected_rows=self.counts)
        with self.assertRaisesRegex(ValueError, "entire source run"):
            build_plan(self.source, self.source.parent / "nested-new-run", expected_rows=self.counts)
        plan = self.plan()
        source_file = Path(plan["files"][0]["source"])
        source_file.write_bytes(source_file.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, "Source changed"):
            copy_inputs(plan)
        self.assertFalse(self.target.exists())
        self.target.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()


if __name__ == "__main__":
    unittest.main()
