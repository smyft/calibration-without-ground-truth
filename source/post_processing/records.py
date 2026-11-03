"""Record parsing and ID alignment shared by post-processing baselines."""


def parse_record(item):
    """Extract confidence, answer, correctness, gold index, and choices."""
    try:
        gold_idx = item["gold"]
        choices = item["choices"]
        answer = item["model_prediction"]["final_answer"]
        confidence = item["model_prediction"]["confidence"]
        try:
            answer_idx = choices.index(answer)
        except ValueError:
            answer_idx = -1
        return confidence, answer, int(answer_idx == gold_idx), gold_idx, choices
    except Exception:
        return 0.0, "", 0, 0, []


def align_records(data1, data2, *, include_reference_correctness=False):
    """Align by ID, preserving primary order and last-reference-record wins."""
    reference_by_id = {item["id"]: item for item in data2}
    records = []
    for item in data1:
        uid = item["id"]
        if uid not in reference_by_id:
            continue
        conf1, answer1, correct1, gold, choices = parse_record(item)
        conf2, answer2, correct2, _, _ = parse_record(reference_by_id[uid])
        record = {
            "id": uid,
            "choices": choices,
            "gold": gold,
            "m1_conf": conf1,
            "m1_pred": answer1,
            "m1_correct": correct1,
            "m2_conf": conf2,
            "m2_pred": answer2,
        }
        if include_reference_correctness:
            record["m2_correct"] = correct2
        records.append(record)
    return records
