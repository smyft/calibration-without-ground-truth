"""Load local prepared datasets while preserving their original record order.

``DATASET_ROOT`` selects a machine's dataset storage without embedding that
location in source files. Dataset preparation is a separate, explicit step;
these loaders never download datasets.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from datasets import load_from_disk

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASETS, DEFAULT_DATASET_ROOT, LOCAL_DATASET_DIRS

MMLU_SUBJECTS = (
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions',
)


def load_local_split(name, subject, split):
    """Read a prepared Arrow split without creating caches or contacting a hub.

    Expected layout: ``<root>/<dataset>/prepared/<configuration>/<split>``.
    ``root`` comes from ``DATASET_ROOT`` or ``DEFAULT_DATASET_ROOT``; CSQA uses
    the ``default`` configuration, while MMLU variants use the given subject.
    """
    root = Path(os.environ.get("DATASET_ROOT", DEFAULT_DATASET_ROOT)).expanduser()
    configuration = "default" if name == "CommonSenseQA" else subject
    path = (root / LOCAL_DATASET_DIRS[name] / "prepared" / configuration / split).resolve()
    required = [path / "state.json", path / "dataset_info.json"]
    if required[0].is_file():
        state = json.loads(required[0].read_text(encoding="utf-8"))
        shards = state.get("_data_files", [])
        if not shards:
            raise ValueError(f"No Arrow shards listed in {required[0]}")
        required.extend(path / shard["filename"] for shard in shards)
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Local dataset {name}/{configuration}/{split} is incomplete at {path}. "
            "Set DATASET_ROOT to the prepared dataset root. No download was attempted. "
            "Missing: " + ", ".join(missing)
        )
    return load_from_disk(str(path))


def prepare_mmlu_dataset(subject: str, split: str = "test") -> List[Dict]:
    """
    Loads a local prepared MMLU split and formats it.

    Args:
        subject: The specific MMLU subject (e.g., 'astronomy').
        split: Dataset split ('test', 'validation', 'dev').

    Returns:
        samples: A list of dictionaries containing formatted prompts and ground truth.
    """
    print(f"Loading MMLU dataset subject: {subject}...")
    try:
        dataset = load_local_split("MMLU", subject, split)

        samples = []
        for i, item in enumerate(dataset):
            samples.append({
                'id': i,
                'choices': item['choices'],
                'gold': item['answer'],
                'task_query': item['question'],
                'subject': item['subject']
            })

        print(f"Loaded {len(samples)} examples.")
        return samples
    except FileNotFoundError:
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def prepare_commonsense_qa(split='validation'):
    """
    Loads a local prepared CommonSenseQA split and formats it.

    Args:
        split: train / validation / test

    Returns:
        samples: A list of dictionaries containing formatted prompts and ground truth.
    """
    try:
        dataset = load_local_split("CommonSenseQA", "default", split)

        samples = []
        for idx, item in enumerate(dataset):
            ans_char = item['answerKey'] 
            choices_labels = item['choices']['label']
            try:
                gold = choices_labels.index(ans_char)
            except ValueError:
                print(f'Label not found for record {idx}')
                gold = -1

            samples.append({
                'id': item.get('id', idx),
                'choices': item['choices']['text'],
                'gold': gold,
                'task_query': item['question'],
                'question_concept': item['question_concept']
            })
        print(f"Successfully loaded {len(samples)} examples in CommonSenseQA dataset.")
        return samples
    except FileNotFoundError:
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
def prepare_mmlu_redux_dataset(subject: str) -> List[Dict]:
    """
    Loads a local prepared MMLU-Redux subject and formats it.

    Args:
        subject: The specific MMLU-Redux subject (e.g., 'astronomy').

    Returns:
        samples: A list of dictionaries containing formatted prompts and ground truth.
    """
    print(f"Loading MMLU-Redux dataset subject: {subject}...")
    try:
        dataset = load_local_split("MMLU_Redux", subject, "test")

        samples = []
        for i, item in enumerate(dataset):
            if item['error_type'] in ["no_correct_answer", "multiple_correct_answers", "wrong_groundtruth"]:
                if item['correct_answer'] in ['0', '1', '2', '3', 0, 1, 2, 3]:
                    gold = item['correct_answer']
                else:
                    gold = item['answer']
            else:
                gold = item['answer']
            samples.append({
                'id': i,
                'choices': item['choices'],
                'gold': int(gold),
                'task_query': item['question'],
                'subject': subject
            })

        print(f"Loaded {len(samples)} examples.")
        return samples
    except FileNotFoundError:
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def prepare_dataset(name, split=None, subject="all", data_file=None):
    """Load local prepared datasets using the original ID/label rules.

    ``data_file`` supplies already formatted records (JSONL) for offline runs.
    Each record must contain id, choices, gold and task_query, plus subject for
    subject-based prompts. No sampling, relabeling or shuffling is performed.
    """
    if name == "MMLU-Redux":
        name = "MMLU_Redux"
    if name not in DATASETS:
        raise ValueError(f"Unsupported dataset {name!r}; choose from {DATASETS}")
    if data_file is not None:
        records = []
        with open(data_file, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                required = {"id", "choices", "gold", "task_query"}
                if name != "CommonSenseQA":
                    required.add("subject")
                missing = required.difference(record)
                if missing:
                    raise ValueError(f"{data_file}:{number}: missing {sorted(missing)}")
                expected_choices = 5 if name == "CommonSenseQA" else 4
                if len(record["choices"]) != expected_choices:
                    raise ValueError(f"{data_file}:{number}: expected {expected_choices} choices")
                records.append(record)
        return records
    if name == "MMLU":
        return prepare_mmlu_dataset(subject, split=split or "test")
    if name == "CommonSenseQA":
        return prepare_commonsense_qa(split=split or "validation")
    if split not in (None, "test"):
        raise ValueError("MMLU_Redux only provides the test split")
    subjects = MMLU_SUBJECTS if subject == "all" else (subject,)
    records = []
    for item in subjects:
        samples = prepare_mmlu_redux_dataset(item)
        if samples is None:
            raise RuntimeError(f"Failed to load MMLU_Redux subject {item}")
        records.extend(samples)
    # Original inference concatenates subjects in this order and assigns global IDs.
    return [dict(record, id=index) for index, record in enumerate(records)]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Inspect formatted dataset records.")
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--split")
    parser.add_argument("--subject", default="all")
    parser.add_argument("--data_file", help="Optional local formatted JSONL input")
    args = parser.parse_args(argv)
    records = prepare_dataset(args.dataset, args.split, args.subject, args.data_file)
    if records is None:
        raise SystemExit("Dataset could not be loaded; check the local prepared files and DATASET_ROOT.")
    print(f"Loaded {len(records)} records")


if __name__ == "__main__":
    main()
