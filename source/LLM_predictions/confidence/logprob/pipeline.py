"""Shared local-model inference for the three supported multiple-choice datasets.

Run from the code directory with ``python -m
source.LLM_predictions.confidence.logprob.pipeline --help``. The dataset-specific
modules remain as thin entry points for existing commands.
"""

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.config import DATASET_CONFIG, MODEL_IDS, get_model_spec, resolve_model_path
from source.LLM_predictions.utils.utils import (
    THINKING_FILTER_VERSION,
    extract_thinking_process,
    load_prompt_templates,
)

PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"
MODES = ("base", "instruct", "thinking_phase1", "thinking_phase2")


def choice_letters(dataset_name, mode="instruct"):
    count = 5 if dataset_name == "CommonSenseQA" else 4
    return tuple(chr(ord("A") + index) for index in range(count))


def extract_logprobs_from_metadata(top_logprobs, choices=("A", "B", "C", "D")):
    """Sum probability mass for whitespace variants of each option token."""
    values = {choice: [] for choice in choices}
    for item in top_logprobs.values():
        token = item.decoded_token
        if token and token.strip() in values:
            values[token.strip()].append(item.logprob)
    return {
        choice: np.logaddexp.reduce(logprobs) if logprobs else -999.0
        for choice, logprobs in values.items()
    }


def score_record(record, top_logprobs, choices):
    scores = extract_logprobs_from_metadata(top_logprobs or {}, choices)
    prediction = max(scores, key=scores.get)
    logits = np.array(list(scores.values()))
    if np.all(logits == -999.0):
        confidence = 1.0 / len(choices)
        prediction = "Unknown"
    else:
        exp_logits = np.exp(logits - np.max(logits))
        confidence = float(np.max(exp_logits / np.sum(exp_logits)))
    return {
        "id": record["id"],
        "choices": list(choices),
        "gold": record["gold"],
        "model_prediction": {
            "final_answer": prediction,
            "confidence": round(confidence, 4),
        },
        "logprobs_raw": scores,
    }


def load_records(dataset_name, split, data_file=None):
    from data.data_loader import prepare_dataset

    return prepare_dataset(dataset_name, split=split, data_file=data_file)


def load_examples(dataset_name, template_name, templates, mode):
    """Keep the original dev/train selection and first-five ordering."""
    examples = defaultdict(list)
    if "example_prompt" not in templates:
        return examples
    if (dataset_name != "CommonSenseQA" or mode == "thinking_phase1") and "5_shot" not in template_name:
        return examples
    example_dataset = "CommonSenseQA" if dataset_name == "CommonSenseQA" else "MMLU"
    example_split = "train" if dataset_name == "CommonSenseQA" else "dev"
    for record in load_records(example_dataset, example_split) or []:
        subject = record.get("subject", "")
        if dataset_name == "CommonSenseQA" or subject:
            examples[subject].append(record)
    return examples


def format_question(template, record, choices):
    values = {f"choice{letter}": record["choices"][index] for index, letter in enumerate(choices)}
    values["task_query"] = record["task_query"]
    return template.format(**values).strip()


def build_prompts(records, templates, dataset_name, mode, examples=None, limit=None):
    """Build the same plain-text prompts or chat messages as the original scripts."""
    examples = examples or {}
    letters = choice_letters(dataset_name)
    prompts, selected = [], []
    for record_index, record in enumerate(records):
        if limit is not None and record_index >= limit:
            break
        subject = record.get("subject", "")
        if (not record.get("task_query") or len(record.get("choices", [])) != len(letters)
                or (dataset_name != "CommonSenseQA" and not subject)):
            print(f"Warning: Skipping record {record_index} due to missing query/choices.")
            continue
        system = templates.get("system_prompt", "")
        if dataset_name != "CommonSenseQA":
            system = system.format(subject=subject)
        system = system.strip()
        few_shot = []
        for example in examples.get(subject if dataset_name != "CommonSenseQA" else "", [])[:5]:
            if len(example.get("choices", [])) == len(letters) and example.get("gold") is not None:
                answer_index = example["gold"]
                answer = dict(enumerate(letters)).get(answer_index, str(answer_index))
                few_shot.append((example, answer))
        main = format_question(templates.get("main_prompt", ""), record, letters)
        if mode == "base":
            formatted_examples = []
            for example, answer in few_shot:
                values = {f"choice{letter}": example["choices"][index] for index, letter in enumerate(letters)}
                values.update(task_query=example["task_query"], answer=answer)
                formatted_examples.append(templates.get("example_prompt", "").format(**values).strip())
            parts = [system, "\n\n".join(formatted_examples), main]
            prompt = "\n\n".join(part for part in parts if part) + "\nAnswer:"
        else:
            prompt = [{"role": "system", "content": system}] if system else []
            if "main_prompt" in templates:
                for example, answer in few_shot:
                    prompt.extend([
                        {"role": "user", "content": format_question(templates["main_prompt"], example, letters)},
                        {"role": "assistant", "content": f"Answer: {answer}"},
                    ])
            prompt.append({"role": "user", "content": main})
            if mode == "instruct":
                prompt.append({"role": "assistant", "content": "Answer:"})
        prompts.append(prompt)
        selected.append(record)
    return prompts, selected


def build_phase2_prompts(records):
    return [
        list(record["conversation"]) + [{
            "role": "assistant",
            "content": f"{record['thinking']}\n\nSo the final answer is (answer with an option letter):",
        }]
        for record in records
    ]


def prediction_path(args, mode):
    # Model IDs and local directory names are deliberately separate: moving a
    # checkpoint must not change the filenames consumed by calibration scripts.
    model_id = get_model_spec(args.model_name).model_id
    directory = (Path(args.output_dir) / args.dataset_name / args.split / "LLMs"
                 / "confidence" / "logprob" / model_id.replace("/", "_"))
    if mode == "thinking_phase1":
        return directory / "intermediate" / f"{args.confidence_prompt_template}_phase1.jsonl"
    return directory / f"{args.confidence_prompt_template}.jsonl"


def sampling_kwargs(args, mode):
    kwargs = {"temperature": args.temperature}
    if mode == "thinking_phase1":
        kwargs.update(max_tokens=getattr(args, "thinking_max_tokens", 1024), skip_special_tokens=False)
    else:
        kwargs.update(max_tokens=args.max_tokens, logprobs=20)
    # Previously the parser accepted --seed but never forwarded it. Omission
    # retains the original vLLM defaults; an explicit seed now takes effect.
    if args.seed is not None:
        kwargs["seed"] = args.seed
    return kwargs


def free_gpu_memory(llm_instance=None):
    """Collect references after the caller has released its engine."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def ministral_hf_overrides(model_path):
    """Translate saved Ministral metadata into vLLM's HF configuration fields.

    The native params retain embedding sharing and long-context scaling values
    that some HF exports omit or encode differently. Apply only those fields
    in memory, keeping the existing text configuration and checkpoint files.
    """
    model_path = Path(model_path)
    config_path = model_path / "config.json"
    params_path = model_path / "params.json"
    if not config_path.is_file() or not params_path.is_file():
        return {}
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    params = json.loads(params_path.read_text(encoding="utf-8"))
    raw_text = raw_config.get("text_config", {})
    tied_embeddings = params.get("tied_embeddings")
    apply_yarn_scaling = params.get("yarn", {}).get("apply_scale")
    llama_4_scaling = params.get("llama_4_scaling")

    def reject_conflict(field, saved, native):
        if saved != native:
            raise ValueError(
                f"Conflicting embedding metadata or scaling metadata in {model_path}: "
                f"config.json {field}={saved!r}, params.json value={native!r}. "
                "Verify the checkpoint configuration before loading."
            )

    if isinstance(tied_embeddings, bool):
        for container, prefix in ((raw_config, ""), (raw_text, "text_config.")):
            if "tie_word_embeddings" in container:
                reject_conflict(prefix + "tie_word_embeddings", container["tie_word_embeddings"], tied_embeddings)
    if isinstance(apply_yarn_scaling, bool):
        raw_rope = raw_text.get("rope_parameters", raw_text.get("rope_scaling", {})) or {}
        if "apply_yarn_scaling" in raw_rope:
            reject_conflict("text_config.rope_parameters.apply_yarn_scaling", raw_rope["apply_yarn_scaling"], apply_yarn_scaling)
    if isinstance(llama_4_scaling, dict) and "llama_4_scaling" in raw_text:
        reject_conflict("text_config.llama_4_scaling", raw_text["llama_4_scaling"], llama_4_scaling)
    if not (isinstance(tied_embeddings, bool) or isinstance(apply_yarn_scaling, bool)
            or isinstance(llama_4_scaling, dict)):
        return {}

    def apply_overrides(config):
        # vLLM probes this callable with a dummy config before loading the
        # actual nested Mistral3Config. Leave that probe untouched.
        text = getattr(config, "text_config", None)
        if text is None:
            return config
        if isinstance(tied_embeddings, bool):
            config.tie_word_embeddings = tied_embeddings
            text.tie_word_embeddings = tied_embeddings
        if isinstance(apply_yarn_scaling, bool):
            rope = getattr(text, "rope_parameters", None)
            if not isinstance(rope, dict):
                raise ValueError(f"Missing text_config.rope_parameters for Ministral checkpoint {model_path}")
            rope["apply_yarn_scaling"] = apply_yarn_scaling
        if isinstance(llama_4_scaling, dict):
            text.llama_4_scaling = dict(llama_4_scaling)
        return config

    return apply_overrides

def run_inference(args, mode):
    """Execute one mode, preserving prompt, sampling, score and JSONL semantics."""
    if mode not in MODES:
        raise ValueError(f"Unsupported inference mode: {mode}")
    if args.dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    spec = get_model_spec(args.model_name)
    expected_variant = "base" if mode == "base" else "instruct"
    if spec.variant != expected_variant:
        raise ValueError(f"Mode {mode!r} requires a {expected_variant} model; got {spec.model_id}.")
    model_path = resolve_model_path(
        args.model_name, model_root=getattr(args, "model_root", None),
        model_path=getattr(args, "model_path", None),
    )
    target = prediction_path(args, mode)
    if target.exists() and not getattr(args, "overwrite", False):
        raise FileExistsError(f"Refusing to overwrite {target}; use a separate --output_dir or --overwrite.")
    limit = getattr(args, "limit", None)
    if args.debug:
        limit = min(limit, 5) if limit is not None else 5
    if mode == "thinking_phase2":
        source = prediction_path(args, "thinking_phase1")
        with source.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        records = records if limit is None else records[:limit]
        prompts = build_phase2_prompts(records)
    else:
        # Redux's direct scoring scripts historically use the MMLU templates.
        prompt_dataset = "MMLU" if args.dataset_name == "MMLU_Redux" and mode != "thinking_phase1" else args.dataset_name
        templates = load_prompt_templates(PROMPT_ROOT / prompt_dataset / "logprob" / args.confidence_prompt_template)
        raw_records = load_records(args.dataset_name, args.split, getattr(args, "data_file", None))
        if not raw_records:
            print("No data loaded. Exiting.")
            return None
        examples = load_examples(args.dataset_name, args.confidence_prompt_template, templates, mode)
        prompts, records = build_prompts(raw_records or [], templates, args.dataset_name, mode, examples, limit)
    if not records:
        print("No data loaded. Exiting.")
        return None

    # Heavy GPU imports happen only when inference is actually requested. CLI
    # help, prompt construction and score tests work without vLLM installed.
    from vllm import LLM, SamplingParams

    engine_kwargs = dict(
        model=str(model_path), tokenizer=str(model_path), trust_remote_code=True,
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
        dtype="auto", enforce_eager=True,
    )
    chat_kwargs = {}
    chat_template = model_path / "chat_template.jinja"
    if spec.family != "ministral" and spec.variant == "instruct" and chat_template.is_file():
        # Some tokenizer backends do not discover the checkpoint's own Jinja
        # file. Supply its original text on each chat call: vLLM does not
        # consistently fall back to the template passed to the constructor.
        chat_kwargs["chat_template"] = chat_template.read_bytes().decode("utf-8")
    if spec.family == "ministral":
        # The native renderer supports the checkpoint's chat protocol; the HF
        # Mistral backend does not implement get_chat_template. Its token IDs
        # match the native renderer for the supported experiment prompts.
        engine_kwargs["tokenizer_mode"] = "mistral"
        # These experiments supply text only. Avoid initializing the optional
        # vision processor or reserving image tokens; text tokenization is unchanged.
        engine_kwargs["language_model_only"] = True
        hf_overrides = ministral_hf_overrides(model_path)
        if hf_overrides:
            engine_kwargs["hf_overrides"] = hf_overrides
    if getattr(args, "max_model_len", None) is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if getattr(args, "tensor_parallel_size", None) is not None:
        engine_kwargs["tensor_parallel_size"] = args.tensor_parallel_size
    llm = None
    try:
        llm = LLM(**engine_kwargs)
        sampling = SamplingParams(**sampling_kwargs(args, mode))
        if mode == "base":
            outputs = llm.generate(prompts, sampling)
        elif mode == "thinking_phase1":
            outputs = llm.chat(prompts, sampling, **chat_kwargs)
        else:
            outputs = llm.chat(prompts, sampling, add_generation_prompt=False, continue_final_message=True, **chat_kwargs)
        rows = []
        for record, prompt, output in zip(records, prompts, outputs):
            if args.debug:
                print(llm.get_tokenizer().decode(output.prompt_token_ids))
                print(output.outputs)
            if mode == "thinking_phase1":
                keys = ("id", "subject", "task_query", "choices", "gold") if args.dataset_name != "CommonSenseQA" else ("id", "task_query", "choices", "gold")
                row = {key: record[key] for key in keys}
                raw_generation = output.outputs[0].text
                legal_options = choice_letters(args.dataset_name, mode)
                thinking = extract_thinking_process(raw_generation, legal_options)
                row.update(
                    thinking=thinking, conversation=prompt,
                    raw_generation=raw_generation,
                    filter_version=THINKING_FILTER_VERSION,
                    legal_options=list(legal_options),
                    filter_diagnostics={
                        "raw_char_count": len(raw_generation),
                        "thinking_char_count": len(thinking),
                        "raw_empty": not bool(raw_generation.strip()),
                        "thinking_empty": not bool(thinking.strip()),
                    },
                )
            else:
                token_logprobs = output.outputs[0].logprobs
                first_token = (token_logprobs[0] if token_logprobs else {}) if mode == "thinking_phase2" else token_logprobs[0]
                row = score_record(record, first_token, choice_letters(args.dataset_name, mode))
            rows.append(row)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Recheck at open time so concurrent jobs cannot replace a result.
        with target.open("w" if getattr(args, "overwrite", False) else "x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=mode != "thinking_phase1") + "\n")
        print(f"Saved {len(rows)} records to {target}")
        return target
    finally:
        try:
            # Explicitly close vLLM's worker processes even when chat rendering
            # or generation fails; deleting the frontend alone can leave them alive.
            engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
            shutdown = getattr(engine_core, "shutdown", None)
            if callable(shutdown):
                shutdown()
        finally:
            del llm
            free_gpu_memory()


def build_parser(dataset_name=None, mode=None):
    parser = argparse.ArgumentParser(description="Score multiple-choice answers using local model checkpoints.")
    parser.add_argument("--dataset_name", choices=tuple(DATASET_CONFIG), default=dataset_name or "MMLU_Redux")
    parser.add_argument("--mode", choices=MODES, default=mode or "instruct")
    parser.add_argument("--split", default=None, help="Defaults to validation for CommonSenseQA and test otherwise.")
    parser.add_argument("--model_name", choices=MODEL_IDS, required=True)
    parser.add_argument("--data_file", help="Optional local JSONL file containing normalized dataset records.")
    parser.add_argument("--model_root", help="Directory containing the local model checkpoints.")
    parser.add_argument("--model_path", help="Explicit local checkpoint directory; never downloaded.")
    parser.add_argument("--confidence_prompt_template", default=None,
                        choices=("baseline", "5_shot", "baseline_thinking"))
    parser.add_argument("-t", "--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=1)
    parser.add_argument("--thinking_max_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--limit", type=int, help="Use at most this many input records.")
    parser.add_argument("--max_model_len", type=int, help="Optional vLLM context length override.")
    parser.add_argument("--tensor_parallel_size", type=int)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--debug", action="store_true", help="Print model input/output and process at most five records.")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing the selected result file.")
    return parser


def main(argv=None, dataset_name=None, mode=None):
    parser = build_parser(dataset_name, mode)
    args = parser.parse_args(argv)
    args.split = args.split or ("validation" if args.dataset_name == "CommonSenseQA" else "test")
    args.confidence_prompt_template = args.confidence_prompt_template or (
        "baseline_thinking" if args.mode.startswith("thinking_") else "5_shot")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return run_inference(args, args.mode)


if __name__ == "__main__":
    main()
