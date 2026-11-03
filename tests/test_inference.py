"""Regression checks for prompt bytes, sampling parameters and JSONL results.

The golden digests were captured from the pre-refactor inference functions with
an in-memory model and deterministic dataset fixtures. No GPU, model downloads,
or existing experiment outputs are used by these tests.
"""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from source.LLM_predictions.confidence.logprob import pipeline
from source.LLM_predictions.utils.utils import (
    THINKING_FILTER_VERSION, extract_thinking_process, filter_answers_from_content,
)


MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512"
DATASETS = ("MMLU", "MMLU_Redux", "CommonSenseQA")
SCENARIOS = tuple(
    (dataset, mode, template)
    for dataset in DATASETS
    for mode, template in (
        ("base", "baseline"), ("base", "5_shot"),
        ("instruct", "baseline"), ("instruct", "5_shot"),
        ("thinking_phase1", "baseline_thinking"),
        ("thinking_phase2", "baseline_thinking"),
    )
)
# Filled from the original entry points, before comparing the shared pipeline.
REFERENCE_DIGESTS = {
    "MMLU/base/baseline": "8dc7671ac01b444eb6c4851b97a053c0be85efe96a9387646c4837778630f67f",
    "MMLU/base/5_shot": "2bd02e7b5080e872b7a1c9110a95caae6d289a5d4dfdc7fedd746089b0959b90",
    "MMLU/instruct/baseline": "bf77bab73f1f47d5043f4ffaf1a6d6a0956c828449985d2ed93a9e5438073e22",
    "MMLU/instruct/5_shot": "f4fd2de0840a82a649d3e5ad25bc642f0f1c9381f65ec70b05b0e943dcb547cb",
    "MMLU/thinking_phase1/baseline_thinking": "3bca316ab7ac41ad8a6d0961fc9e4ebb8c14ad94aec7cd9182d3afdc3ad570c7",
    "MMLU/thinking_phase2/baseline_thinking": "94517062e4afc8fd6882b825907b2a81ceca73b3cf6baf10903665e3135cab68",
    "MMLU_Redux/base/baseline": "8dc7671ac01b444eb6c4851b97a053c0be85efe96a9387646c4837778630f67f",
    "MMLU_Redux/base/5_shot": "2bd02e7b5080e872b7a1c9110a95caae6d289a5d4dfdc7fedd746089b0959b90",
    "MMLU_Redux/instruct/baseline": "bf77bab73f1f47d5043f4ffaf1a6d6a0956c828449985d2ed93a9e5438073e22",
    "MMLU_Redux/instruct/5_shot": "f4fd2de0840a82a649d3e5ad25bc642f0f1c9381f65ec70b05b0e943dcb547cb",
    "MMLU_Redux/thinking_phase1/baseline_thinking": "3bca316ab7ac41ad8a6d0961fc9e4ebb8c14ad94aec7cd9182d3afdc3ad570c7",
    "MMLU_Redux/thinking_phase2/baseline_thinking": "94517062e4afc8fd6882b825907b2a81ceca73b3cf6baf10903665e3135cab68",
    "CommonSenseQA/base/baseline": "b209d80ff7229ad93c74fcd76b09a32ccf6e0d7439ba8e4995a6ac7f56886848",
    "CommonSenseQA/base/5_shot": "ebda1814025f899fc61f33ab206f39d69c33f671940619c608f0d8c3272b0b50",
    "CommonSenseQA/instruct/baseline": "c3006a0b1992f480ff3d86eb90178ce8b6e8540d9daf49a82f3b49d7433c7d64",
    "CommonSenseQA/instruct/5_shot": "92c34dcdda39d18abc49c830854214e7ab134353fcbe7b4a1d15e72eec13694d",
    "CommonSenseQA/thinking_phase1/baseline_thinking": "deae161d9905885437abbc344a11914e91a22abd81b5a968befdb9502276c8e8",
    "CommonSenseQA/thinking_phase2/baseline_thinking": "e87f230604b12df34cde76dfda14f722e368980d994d9e8ebefa0c57819b6301"
}


def fixture_records(dataset, examples=False):
    count = 5 if dataset == "CommonSenseQA" else 4
    records = []
    for index in range(9):
        record = {
            "id": index, "task_query": f"  Question {index}: α?  ",
            "choices": [f"Choice {letter} {index}" for letter in "ABCDE"[:count]],
            "gold": index % count,
        }
        if dataset != "CommonSenseQA":
            record["subject"] = "abstract_algebra" if index != 8 else "anatomy"
        if index == 2:
            record["choices"] = ["invalid"]
        if index == 3 and not examples:
            record["task_query"] = ""
        if index == 4 and dataset != "CommonSenseQA" and not examples:
            record["subject"] = ""
        if index == 1 and examples:
            record["gold"] = None
        records.append(record)
    return records


def fixture_load(dataset, split, data_file=None):
    return fixture_records(dataset, examples=split in ("train", "dev"))


def fake_output(index):
    token_sets = (
        [(" A", -2.0), ("A", -3.0), ("B", -2.5), ("C", -4.0), ("D", -5.0), (" E", -1.0), (None, -1.0)],
        [("invalid", -1.0)],
        [(letter, -2.0) for letter in "ABCDE"],
    )
    logprobs = {
        position: SimpleNamespace(decoded_token=token, logprob=value)
        for position, (token, value) in enumerate(token_sets[index % len(token_sets)])
    }
    thinking = (
        "<think>Compare the quantities. Check consistency. The answer is A. This follows.</think>\nA",
        "<think>Use the definition. Compute the total.",
        "Begin with a relation. Check its units. Therefore, the answer is B.",
    )[index % 3]
    return SimpleNamespace(outputs=[SimpleNamespace(logprobs=[logprobs], text=thinking)], prompt_token_ids=[1, 2])


class FakeSampling:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    calls = []
    engine_kwargs = []

    def __init__(self, **kwargs):
        type(self).engine_kwargs.append(kwargs)

    def generate(self, prompts, sampling_params):
        return self._record("generate", prompts, sampling_params, {})

    def chat(self, prompts, sampling_params, **kwargs):
        return self._record("chat", prompts, sampling_params, kwargs)

    def _record(self, method, prompts, sampling, kwargs):
        type(self).calls.append([method, copy.deepcopy(prompts), sampling.kwargs, kwargs])
        return [fake_output(index) for index in range(len(prompts))]

    def get_tokenizer(self):
        return SimpleNamespace(decode=lambda ids: "fake prompt")


def make_args(directory, dataset, mode, template):
    return SimpleNamespace(
        output_dir=str(directory), model_name=MODEL_ID.replace("Instruct", "Base") if mode == "base" else MODEL_ID, dataset_name=dataset,
        split="validation" if dataset == "CommonSenseQA" else "test",
        confidence_prompt_template=template, temperature=0, seed=None,
        max_tokens=1, debug=False, mode=mode,
    )


def prepare_phase2_input(args):
    target = pipeline.prediction_path(args, "thinking_phase1")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in fixture_records(args.dataset_name)[:5]:
            record.update(thinking="Consider the alternatives.", conversation=[
                {"role": "system", "content": "System"},
                {"role": "user", "content": record["task_query"]},
            ])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def payload_digest(args, mode):
    target = pipeline.prediction_path(args, mode)
    output = target.read_text(encoding="utf-8")
    if mode == "thinking_phase1":
        # The frozen golden covers original inference behavior. Audit fields are
        # checked separately and do not change the original record payload.
        rows = [json.loads(line) for line in output.splitlines()]
        for row in rows:
            for key in ("raw_generation", "filter_version", "legal_options", "filter_diagnostics"):
                row.pop(key)
        output = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    payload = {"calls": FakeLLM.calls, "output": output}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


class InferenceRegressionTests(unittest.TestCase):
    def test_prompt_sampling_and_result_equivalence(self):
        for dataset, mode, template in SCENARIOS:
            with self.subTest(dataset=dataset, mode=mode, template=template):
                with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                    args = make_args(directory, dataset, mode, template)
                    if mode == "thinking_phase2":
                        prepare_phase2_input(args)
                    FakeLLM.calls = []
                    FakeLLM.engine_kwargs = []
                    with patch.object(pipeline, "load_records", side_effect=fixture_load), \
                            patch.object(pipeline, "resolve_model_path", return_value=Path(directory)), \
                            patch.object(pipeline, "free_gpu_memory"), \
                            patch.dict("sys.modules", {"vllm": SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSampling)}), \
                            contextlib.redirect_stdout(io.StringIO()):
                        pipeline.run_inference(args, mode)
                    key = "/".join((dataset, mode, template))
                    self.assertEqual(payload_digest(args, mode), REFERENCE_DIGESTS[key])
                    if mode == "thinking_phase1":
                        rows = [json.loads(line) for line in pipeline.prediction_path(args, mode).read_text().splitlines()]
                        for index, row in enumerate(rows):
                            raw = fake_output(index).outputs[0].text
                            self.assertEqual(row["raw_generation"], raw)
                            self.assertEqual(row["filter_version"], THINKING_FILTER_VERSION)
                            self.assertEqual(row["legal_options"], list(pipeline.choice_letters(dataset)))
                            self.assertEqual(row["filter_diagnostics"], {
                                "raw_char_count": len(raw), "thinking_char_count": len(row["thinking"]),
                                "raw_empty": not bool(raw.strip()), "thinking_empty": not bool(row["thinking"].strip()),
                            })
                    self.assertEqual(FakeLLM.engine_kwargs[0]["model"], directory)
                    self.assertEqual(FakeLLM.engine_kwargs[0]["tokenizer"], directory)
                    self.assertEqual({key: value for key, value in FakeLLM.engine_kwargs[0].items()
                                      if key not in ("model", "tokenizer")}, {
                        "trust_remote_code": True, "gpu_memory_utilization": 0.9,
                        "dtype": "auto", "enforce_eager": True, "language_model_only": True,
                        "tokenizer_mode": "mistral",
                    })

    def test_ministral_missing_hf_flag_uses_saved_embedding_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text("{}")
            (model_path / "params.json").write_text('{"tied_embeddings": false}')
            before = {name: (model_path / name).read_bytes() for name in ("config.json", "params.json")}
            override = pipeline.ministral_hf_overrides(model_path)
            config = SimpleNamespace(tie_word_embeddings=True, text_config=SimpleNamespace(tie_word_embeddings=False))
            self.assertIs(override(config), config)
            self.assertFalse(config.tie_word_embeddings)
            args = make_args(directory, "MMLU", "base", "baseline")
            FakeLLM.engine_kwargs = []
            with patch.object(pipeline, "load_records", side_effect=fixture_load), \
                    patch.object(pipeline, "resolve_model_path", return_value=model_path), \
                    patch.object(pipeline, "free_gpu_memory"), \
                    patch.dict("sys.modules", {"vllm": SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSampling)}), \
                    contextlib.redirect_stdout(io.StringIO()):
                pipeline.run_inference(args, "base")
            self.assertTrue(callable(FakeLLM.engine_kwargs[0]["hf_overrides"]))
            self.assertEqual({name: (model_path / name).read_bytes() for name in before}, before)

    def test_ministral_explicit_embedding_flag_is_not_overridden(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text('{"tie_word_embeddings": false}')
            (model_path / "params.json").write_text('{"tied_embeddings": false}')
            override = pipeline.ministral_hf_overrides(model_path)
            config = SimpleNamespace(tie_word_embeddings=False, text_config=SimpleNamespace(tie_word_embeddings=False))
            before = copy.deepcopy(config)
            self.assertIs(override(config), config)
            self.assertEqual(config, before)

    def test_ministral_conflicting_embedding_flags_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text('{"tie_word_embeddings": true}')
            (model_path / "params.json").write_text('{"tied_embeddings": false}')
            with self.assertRaisesRegex(ValueError, "Conflicting embedding metadata"):
                pipeline.ministral_hf_overrides(model_path)

    def test_ministral_native_scaling_changes_only_saved_fields(self):
        native = {
            "tied_embeddings": False,
            "yarn": {"original_max_position_embeddings": 16384, "factor": 16,
                     "apply_scale": False, "beta": 32, "alpha": 1},
            "llama_4_scaling": {"original_max_position_embeddings": 16384, "beta": 0.1},
        }
        rope = {
            "beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0,
            "llama_4_scaling_beta": 0.1, "mscale": 1.0, "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 16384, "rope_theta": 1000000.0,
            "rope_type": "yarn", "type": "yarn",
        }
        for variant in ("base", "instruct"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
                model_path = Path(directory)
                raw = {"text_config": {"tie_word_embeddings": False, "rope_parameters": rope}}
                if variant == "instruct":
                    raw["tie_word_embeddings"] = False
                (model_path / "config.json").write_text(json.dumps(raw))
                (model_path / "params.json").write_text(json.dumps(native))
                files_before = {name: (model_path / name).read_bytes() for name in ("config.json", "params.json")}
                adapter = pipeline.ministral_hf_overrides(model_path)
                dummy = SimpleNamespace(architectures=[""], model_type="dummy_mistral3")
                dummy_before = copy.deepcopy(dummy)
                self.assertIs(adapter(dummy), dummy)
                self.assertEqual(dummy, dummy_before)
                config = SimpleNamespace(
                    tie_word_embeddings=variant == "base", unrelated="preserved",
                    text_config=SimpleNamespace(tie_word_embeddings=False,
                                                rope_parameters=copy.deepcopy(rope), hidden_size=4096),
                )
                expected = copy.deepcopy(config)
                expected.tie_word_embeddings = False
                expected.text_config.rope_parameters["apply_yarn_scaling"] = False
                expected.text_config.llama_4_scaling = dict(native["llama_4_scaling"])
                original_text = config.text_config
                self.assertIs(adapter(config), config)
                self.assertIs(config.text_config, original_text)
                self.assertEqual(config, expected)
                self.assertEqual({name: (model_path / name).read_bytes() for name in files_before}, files_before)

    def test_ministral_explicit_scaling_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(json.dumps({
                "text_config": {"rope_parameters": {"apply_yarn_scaling": True}},
            }))
            (model_path / "params.json").write_text(json.dumps({"yarn": {"apply_scale": False}}))
            with self.assertRaisesRegex(ValueError, "Conflicting.*scaling metadata"):
                pipeline.ministral_hf_overrides(model_path)

    def test_local_chat_template_is_used_only_for_instruct(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            template = Path(directory) / "chat_template.jinja"
            template_bytes = b"{{ messages }}\r\n{# Original local template #}"
            template.write_bytes(template_bytes)
            for mode in pipeline.MODES:
                with self.subTest(mode=mode):
                    prompt_template = "baseline_thinking" if mode.startswith("thinking_") else "baseline"
                    args = make_args(directory, "MMLU", mode, prompt_template)
                    args.model_name = "Qwen/Qwen3-8B-Base" if mode == "base" else "Qwen/Qwen3-8B"
                    FakeLLM.engine_kwargs = []
                    FakeLLM.calls = []
                    with patch.object(pipeline, "load_records", side_effect=fixture_load), \
                            patch.object(pipeline, "resolve_model_path", return_value=Path(directory)), \
                            patch.object(pipeline, "free_gpu_memory"), \
                            patch.dict("sys.modules", {"vllm": SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSampling)}), \
                            contextlib.redirect_stdout(io.StringIO()):
                        pipeline.run_inference(args, mode)
                    kwargs = FakeLLM.engine_kwargs[0]
                    self.assertNotIn("chat_template", kwargs)
                    call_kwargs = FakeLLM.calls[0][3]
                    if mode == "base":
                        self.assertNotIn("chat_template", call_kwargs)
                    else:
                        self.assertEqual(call_kwargs["chat_template"].encode("utf-8"), template_bytes)
                    self.assertNotIn("tokenizer_mode", kwargs)
                    self.assertEqual(template.read_bytes(), template_bytes)

    def test_ministral_uses_native_renderer_without_custom_template(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            (Path(directory) / "chat_template.jinja").write_text("{{ messages }}")
            for mode in pipeline.MODES:
                with self.subTest(mode=mode):
                    prompt_template = "baseline_thinking" if mode.startswith("thinking_") else "baseline"
                    args = make_args(directory, "MMLU", mode, prompt_template)
                    FakeLLM.engine_kwargs = []
                    FakeLLM.calls = []
                    with patch.object(pipeline, "load_records", side_effect=fixture_load), \
                            patch.object(pipeline, "resolve_model_path", return_value=Path(directory)), \
                            patch.object(pipeline, "free_gpu_memory"), \
                            patch.dict("sys.modules", {"vllm": SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSampling)}), \
                            contextlib.redirect_stdout(io.StringIO()):
                        pipeline.run_inference(args, mode)
                    self.assertEqual(FakeLLM.engine_kwargs[0]["tokenizer_mode"], "mistral")
                    self.assertNotIn("chat_template", FakeLLM.engine_kwargs[0])
                    self.assertNotIn("chat_template", FakeLLM.calls[0][3])

    def test_engine_shutdown_on_generation_failure(self):
        engine_core = SimpleNamespace(shutdown=Mock())

        class FailingLLM(FakeLLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.llm_engine = SimpleNamespace(engine_core=engine_core)

            def chat(self, *args, **kwargs):
                raise RuntimeError("chat rendering failed")

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            args = make_args(directory, "MMLU", "instruct", "baseline")
            with patch.object(pipeline, "load_records", side_effect=fixture_load), \
                    patch.object(pipeline, "resolve_model_path", return_value=Path(directory)), \
                    patch.object(pipeline, "free_gpu_memory") as free_memory, \
                    patch.dict("sys.modules", {"vllm": SimpleNamespace(LLM=FailingLLM, SamplingParams=FakeSampling)}), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "chat rendering failed"):
                    pipeline.run_inference(args, "instruct")
            engine_core.shutdown.assert_called_once_with()
            free_memory.assert_called_once_with()
            self.assertFalse(pipeline.prediction_path(args, "instruct").exists())

    def test_explicit_seed_and_thinking_token_limit(self):
        args = make_args("unused", "MMLU", "thinking_phase1", "baseline_thinking")
        args.seed = 123
        args.thinking_max_tokens = 17
        self.assertEqual(pipeline.sampling_kwargs(args, "thinking_phase1"), {
            "temperature": 0, "max_tokens": 17, "skip_special_tokens": False, "seed": 123,
        })

    def test_phase2_preserves_existing_five_choice_scoring(self):
        result = pipeline.score_record({"id": 1, "gold": 4}, {}, pipeline.choice_letters("CommonSenseQA", "thinking_phase2"))
        self.assertEqual(result["choices"], list("ABCDE"))
        self.assertEqual(result["model_prediction"], {"final_answer": "Unknown", "confidence": 0.2})

    def test_existing_results_are_protected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            args = make_args(directory, "MMLU", "instruct", "baseline")
            target = pipeline.prediction_path(args, "instruct")
            target.parent.mkdir(parents=True)
            target.write_text("unchanged")
            with patch.object(pipeline, "resolve_model_path", return_value=Path(directory)):
                with self.assertRaises(FileExistsError):
                    pipeline.run_inference(args, "instruct")
            self.assertEqual(target.read_text(), "unchanged")

    def test_phase2_does_not_mutate_saved_conversation(self):
        record = {"thinking": "Thought", "conversation": [{"role": "user", "content": "Question"}]}
        original = copy.deepcopy(record)
        prompts = pipeline.build_phase2_prompts([record])
        self.assertEqual(record, original)
        self.assertEqual(prompts[0][-1]["content"], "Thought\n\nSo the final answer is (answer with an option letter):")

    def test_phase2_ignores_raw_generation_and_filter_audit_fields(self):
        original = {"thinking": "", "conversation": [{"role": "user", "content": "Question"}]}
        enriched = dict(original, raw_generation="E. Hidden raw text", filter_version="future",
                        legal_options=list("ABCDE"), filter_diagnostics={"thinking_empty": True})
        self.assertEqual(pipeline.build_phase2_prompts([original]), pipeline.build_phase2_prompts([enriched]))
        self.assertIn("So the final answer is", pipeline.build_phase2_prompts([enriched])[0][-1]["content"])

    def test_model_variant_is_checked_before_loading(self):
        args = make_args("unused", "MMLU", "instruct", "baseline")
        with patch.object(pipeline, "resolve_model_path") as resolve:
            with self.assertRaisesRegex(ValueError, "requires a base model"):
                pipeline.run_inference(args, "base")
            resolve.assert_not_called()

    def test_cli_uses_real_arguments(self):
        args = pipeline.build_parser().parse_args([
            "--dataset_name", "CommonSenseQA", "--model_name", MODEL_ID,
            "--mode", "thinking_phase1", "--limit", "2", "--temperature", "0.7",
        ])
        self.assertEqual((args.dataset_name, args.mode, args.limit, args.temperature),
                         ("CommonSenseQA", "thinking_phase1", 2, 0.7))


class ThinkingFilterTests(unittest.TestCase):
    def test_frozen_four_option_behavior(self):
        corpus = ["", "   ", "Energy is useful. Everyone agrees.",
                  "The answer seems self evident.", "</think>Loose closing tag.",
                  "<think>First.</think><think>Second.</think>"]
        for letter in "ABCD":
            for option in (letter, letter.lower(), f"({letter})", f"[{letter}]", f"**{letter}**",
                           rf"\boxed{{{letter}}}", f"{letter}longword"):
                for pattern in (
                    "The answer is {option}. A description. Keep the reasoning.",
                    "{option}. A description. Keep the reasoning.",
                    "Reasoning first. {option} Last detail.",
                    "Reasoning first. Thus the answer is {option}. Final detail.",
                    "<think>First reason. The answer is {option}.</think>Outside text.",
                    "<THINK>{option}. Next idea. Final reasoning.",
                ):
                    corpus.append(pattern.format(option=option))
        # Captured from the unchanged pre-revision implementation, including its
        # aggressive ordinary-word behavior; no historical regex fixes are folded in.
        expected = "295651adf19104533cfc4f775243ec42fa37e835e82308a3275f32ec04b0ec0d"
        for options in (None, tuple("ABCD")):
            values = [extract_thinking_process(text) if options is None else
                      extract_thinking_process(text, options) for text in corpus]
            self.assertEqual(hashlib.sha256(json.dumps(values).encode()).hexdigest(), expected)

    def test_fifth_option_uses_existing_answer_rules(self):
        for option in "ABCDEabcde":
            for text in (f"{option}. A description. Keep the reasoning.",
                         f"<think>{option}. A description. Keep the reasoning.</think>",
                         f"<THINK>{option}. A description. Keep the reasoning."):
                self.assertEqual(extract_thinking_process(text, "ABCDE"), "Keep the reasoning.")
            for marker in (f"({option})", f"[{option}]", f"**{option}**", rf"\boxed{{{option}}}"):
                self.assertEqual(extract_thinking_process(f"First reason. {marker} Last detail.", "ABCDE"),
                                 "First reason.")
        self.assertEqual(extract_thinking_process("First reason. (E) Last detail.", "ABCD"),
                         "First reason. (E) Last detail.")
        self.assertEqual(extract_thinking_process("E.", "ABCDE"), "")
        self.assertEqual(extract_thinking_process("", "ABCDE"), "")

    def test_new_e_support_does_not_match_inside_words(self):
        for text in ("Energy is useful. Everyone agrees.", "Eel. Eels sleep.",
                     "Some (Evergreen) trees.", "These [elements] persist."):
            self.assertEqual(filter_answers_from_content(text, "ABCDE"),
                             filter_answers_from_content(text, "ABCD"))


if __name__ == "__main__":
    unittest.main()
