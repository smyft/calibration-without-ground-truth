"""Deterministic degraded-reference controls shared by all experiment runners."""

import copy
import hashlib
import numpy as np

DUMMY_RANDOM_SLUG = "dummy_random_ref"
DUMMY_SHUFFLED_RANDOM_SLUG = "dummy_shuffled_random_ref"
DUMMY_TPL = "baseline"

def dummy_calibrated_slug(base_slug: str) -> str:
    return f"dummy_calibrated_ref_{base_slug}"

def shuffled_inst_slug(inst_slug: str) -> str:
    return f"{inst_slug}_shuffled"

def _id_to_int(question_id) -> int:
    if isinstance(question_id, int):
        return question_id
    return int(hashlib.md5(str(question_id).encode()).hexdigest(), 16) % (2 ** 31)

def _get_shuffle_perm(question_id, K: int, base_seed: int = 100) -> list:
    rng = np.random.default_rng(base_seed + _id_to_int(question_id))
    perm = list(range(K))
    rng.shuffle(perm)
    return perm

def apply_choice_shuffle(record: dict, perm: list) -> dict:
    rec = copy.deepcopy(record)
    choices = rec["choices"]
    K = len(choices)
    inv_perm = [0] * K
    for new_idx, old_idx in enumerate(perm):
        inv_perm[old_idx] = new_idx
    rec["gold"] = inv_perm[rec["gold"]]
    old_ans = rec["model_prediction"]["final_answer"]
    if old_ans in choices:
        rec["model_prediction"]["final_answer"] = choices[inv_perm[choices.index(old_ans)]]
    if "logprobs_raw" in rec and rec["logprobs_raw"]:
        old_lp = rec["logprobs_raw"]
        rec["logprobs_raw"] = {
            choices[new_idx]: old_lp[choices[perm[new_idx]]]
            for new_idx in range(K)
        }
    return rec

def make_random_ref(source_records, seed=42):
    rng = np.random.default_rng(seed)
    result = []
    for rec in source_records:
        choices = rec["choices"]
        K = len(choices)
        result.append({
            "id": rec["id"], "choices": choices, "gold": rec["gold"],
            "model_prediction": {
                "final_answer": choices[int(rng.integers(0, K))],
                "confidence":   round(1.0 / K, 6),
            },
        })
    return result

def make_calibrated_ref(source_records, base_calib_records, base_acc):
    result = []
    for src, base in zip(source_records, base_calib_records):
        result.append({
            "id": src["id"], "choices": src["choices"], "gold": src["gold"],
            "model_prediction": {
                "final_answer": base["model_prediction"]["final_answer"],
                "confidence":   round(float(base_acc), 6),
            },
        })
    return result

def make_shuffled_instruct(records: list, base_seed: int = 100) -> list:
    result = []
    for rec in records:
        perm = _get_shuffle_perm(rec["id"], len(rec["choices"]), base_seed)
        result.append(apply_choice_shuffle(rec, perm))
    return result

def make_shuffled_random_ref(source_records: list, base_seed: int = 100) -> list:
    result = []
    for rec in source_records:
        choices = rec["choices"]
        K = len(choices)
        perm = _get_shuffle_perm(rec["id"], K, base_seed)
        inv_perm = [0] * K
        for new_idx, old_idx in enumerate(perm):
            inv_perm[old_idx] = new_idx
        rng_ans = np.random.default_rng(base_seed + _id_to_int(rec["id"]) + 10000)
        result.append({
            "id": rec["id"], "choices": choices, "gold": inv_perm[rec["gold"]],
            "model_prediction": {
                "final_answer": choices[int(rng_ans.integers(0, K))],
                "confidence":   round(1.0 / K, 6),
            },
        })
    return result
