import argparse
import json
import os
from source.post_processing.cli import add_prediction_arguments
from source.post_processing.records import parse_record, align_records
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
from source.post_processing.utils import (load_data, load_confidence_results,
                                          calculate_joint_distribution_for_confidence)

# ==========================================
# 1. Baseline Classes
# ==========================================

class HistogramBinning:
    """
    Baseline 1: Histogram Binning.
    Divides confidence scores into bins and assigns the 
    average empirical accuracy of that bin as the new score.
    """
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_scores = np.zeros(n_bins)

    def fit(self, confidences, labels):
        confidences = np.array(confidences)
        labels = np.array(labels)
        
        for i in range(self.n_bins):
            mask = (confidences > self.bin_boundaries[i]) & (confidences <= self.bin_boundaries[i+1])
            if np.sum(mask) > 0:
                self.bin_scores[i] = np.mean(labels[mask])
            else:
                # Fallback to center of bin if empty
                self.bin_scores[i] = (self.bin_boundaries[i] + self.bin_boundaries[i+1]) / 2

    def predict(self, confidences):
        confidences = np.array(confidences)
        indices = np.digitize(confidences, self.bin_boundaries, right=True) - 1
        # Clip indices to handle edge cases (like 0.0)
        indices = np.clip(indices, 0, self.n_bins - 1)
        return self.bin_scores[indices]
    

class TemperatureScaling:
    def __init__(self):
        self.temperature = 1.0 # Initial temperature T=1

    def fit(self, logits_list, label_indices):
        """
        Fits the optimal temperature using Negative Log Likelihood (NLL).

        Args:
            logits_list: A list where each element is a numpy array representing 
                         the logits for all options of a specific sample.
                         (Note: logprobs returned by LLM APIs can also be treated as Logits)
            label_indices: A list or array of indices (integers) corresponding 
                           to the correct option for each sample.
        """
        
        # Define the objective function: Negative Log Likelihood (NLL)
        def nll_loss(t):
            loss = 0.0
            for i, logits in enumerate(logits_list):
                # Scale the logits
                scaled_logits = logits / t
                # Calculate Log Softmax
                # Formula: log_softmax(x_i) = x_i - log(sum(exp(x_j)))
                log_probs = log_softmax(scaled_logits)
                
                # Retrieve the log_prob of the correct category
                target_idx = label_indices[i]
                loss -= log_probs[target_idx] # NLL = -log(p_target)
            return loss

        # Use Scipy to optimize the scalar T
        # T is usually in the range (0, 1e5] and must be greater than 0
        res = minimize_scalar(nll_loss, bounds=(0.01, 1e5), method='bounded')
        self.temperature = res.x
        print(f"Optimal Temperature found: {self.temperature:.4f}")

    def predict(self, logits):
        """ Calibrate logits for a single sample and return the calibrated probability distribution """
        scaled_logits = np.array(logits) / self.temperature
        return softmax(scaled_logits)


# ==========================================
# 2. Missing-Rate Diagnostic for Our Method
# ==========================================

def compute_missing_rate(results_dir,
                         inst_slug, base_slug,
                         instruct_tpl="baseline_thinking",
                         base_tpl="baseline",
                         calib_dataset="MMLU", calib_split="validation",
                         test_dataset="MMLU",  test_split="test",
                         precision=2):
    """
    Compute the missing rate of our Bregman-projection recalibration map for
    one (instruct, base) model pair.

    A test record is "missing" when its instruct report r=(j, m) — predicted
    class j and discretised confidence m — does not appear in the calibration
    support.  In that case our map cannot provide a recalibrated prediction, so
    the original prediction is kept unchanged.

    Returns a dict with keys:
        n_calib        — number of calibration records
        n_test         — total test records
        n_miss         — test records outside calibration support
        missing_rate   — n_miss / n_test
        n_r1_calib     — unique instruct (j,m) reports in calibration set
    or None if any data file is missing.
    """
    def _path(dataset, split, slug, tpl):
        return os.path.join(results_dir, dataset, split,
                            "LLMs", "confidence", "logprob",
                            slug, f"{tpl}.jsonl")

    calib_inst = load_confidence_results(_path(calib_dataset, calib_split, inst_slug, instruct_tpl))
    calib_base = load_confidence_results(_path(calib_dataset, calib_split, base_slug, base_tpl))
    test_inst  = load_confidence_results(_path(test_dataset,  test_split,  inst_slug, instruct_tpl))
    test_base  = load_confidence_results(_path(test_dataset,  test_split,  base_slug, base_tpl))

    for name, data in [("calib_inst", calib_inst), ("calib_base", calib_base),
                       ("test_inst",  test_inst),  ("test_base",  test_base)]:
        if not data:
            print(f"  [SKIP] Missing data: {name}")
            return None

    # Build joint distribution on calibration split
    jd_calib = calculate_joint_distribution_for_confidence(
        calib_inst, calib_base, precision=precision)
    if jd_calib is None:
        print("  [SKIP] Failed to build calibration joint distribution.")
        return None

    # Build joint distribution on test split (used only to enumerate test reports)
    jd_test = calculate_joint_distribution_for_confidence(
        test_inst, test_base, precision=precision)
    if jd_test is None:
        print("  [SKIP] Failed to build test joint distribution.")
        return None

    # Calibration support: set of instruct (j, m) reports seen during calibration
    calib_support = set(zip(jd_calib["r1_j"].tolist(), jd_calib["r1_m"].tolist()))

    # Test reports: the (j, m) pair for each test record (r1 side of report_pairs)
    test_reports = [p[0] for p in jd_test["report_pairs"]]
    n_test  = len(test_reports)
    n_miss  = sum(1 for r in test_reports if r not in calib_support)
    missing_rate = n_miss / n_test if n_test > 0 else float("nan")

    return {
        "inst_slug":    inst_slug,
        "base_slug":    base_slug,
        "calib_dataset": calib_dataset,
        "calib_split":   calib_split,
        "test_dataset":  test_dataset,
        "test_split":    test_split,
        "n_calib":       len(jd_calib["report_pairs"]),
        "n_r1_calib":    int(jd_calib["grid_stats"]["n_r1"]),
        "n_test":        n_test,
        "n_miss":        n_miss,
        "missing_rate":  missing_rate,
    }


# ==========================================
# 3. Pipeline Functions
# ==========================================

def prepare_data(data1, data2):
    """Align predictions by ID while preserving primary record order."""
    return align_records(data1, data2, include_reference_correctness=False)


def _binary_logits(confidences, eps=1e-12):
    """Represent confidence as [P(correct), P(incorrect)] log probabilities."""
    logits = []
    for confidence in confidences:
        probability = float(np.clip(confidence, eps, 1 - eps))
        logits.append(np.array([np.log(probability), np.log(1 - probability)]))
    return logits


def _calibrate_confidences(method, train_conf, train_correct, test_conf, debug=False):
    if method == "histogram_binning":
        model = HistogramBinning(n_bins=10)
        model.fit(train_conf, train_correct)
        if debug:
            for i in range(model.n_bins):
                print(f"Range ({model.bin_boundaries[i]}, {model.bin_boundaries[i+1]}]: {model.bin_scores[i]}")
        return model.predict(test_conf)
    if method == "isotonic_regression":
        model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        model.fit(train_conf, train_correct)
        return model.predict(test_conf)
    if method == "temperature_scaling":
        model = TemperatureScaling()
        model.fit(_binary_logits(train_conf), [0 if correct else 1 for correct in train_correct])
        confidences = []
        for logits in _binary_logits(test_conf):
            probabilities = model.predict(logits)
            confidences.append(probabilities[0])
            if debug:
                print(f"Original Conf -> Logits: {logits}")
                print(f"Calibrated Probs: {probabilities}, New Conf: {probabilities[0]}")
        return confidences
    raise ValueError(f"Unknown baseline method: {method}")


def process_baselines(args):
    """Fit a confidence-only baseline and retain the primary model's answers."""
    confidence_pairs = load_data(args)
    model1_path_name = args.model1_name.replace("/", "_")
    primary_path = os.path.join(args.model1_method, model1_path_name)
    reference_path = args.model2_name.replace("/", "_") if args.model2_name else "unknown_model2"
    output_dir = os.path.join(
        args.output_dir, args.dataset_name, args.split, "post_processing",
        args.baseline_method, "confidence", primary_path,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    # Preserve the original training-data root, which is --output_dir.
    training_dir = os.path.join(
        args.output_dir, args.aux_dataset, args.aux_split, "LLMs", "confidence",
        "logprob", model1_path_name,
    )
    for (name1, name2), (data1, data2) in confidence_pairs.items():
        if "baseline" not in name1 or "baseline" not in name2:
            continue
        if args.debug:
            data1, data2 = data1[:5], data2[:5]
        records = prepare_data(data1, data2)
        if not records:
            print("No matching records found. Skipping.")
            continue
        train_data = load_confidence_results(os.path.join(training_dir, f"{name1}.jsonl"))
        if not train_data:
            print("No calibration data available. Skipping.")
            continue
        train_values = [parse_record(item) for item in train_data]
        new_confidences = _calibrate_confidences(
            args.baseline_method,
            [values[0] for values in train_values],
            [values[2] for values in train_values],
            [record["m1_conf"] for record in records],
            debug=args.debug,
        )
        output_path = os.path.join(output_dir, f"{name1}_{reference_path}_{name2}.jsonl")
        with open(output_path, "w", encoding="utf-8") as handle:
            for record, confidence in zip(records, new_confidences):
                result = {
                    "id": record["id"], "choices": record["choices"], "gold": record["gold"],
                    "model_prediction": {
                        "final_answer": record["m1_pred"],
                        "confidence": round(float(confidence), 4),
                    },
                }
                handle.write(json.dumps(result) + "\n")
        print(f"Saved {len(records)} processed records to {output_path}")


def build_parser():
    from src.config import DATASET_CONFIG

    parser = add_prediction_arguments(argparse.ArgumentParser(
        description="Fit supervised confidence baselines on a separate calibration split."
    ))
    parser.add_argument("--baseline_method", required=True,
                        choices=("histogram_binning", "isotonic_regression", "temperature_scaling"))
    parser.add_argument("--aux_dataset", choices=tuple(DATASET_CONFIG), default="MMLU")
    parser.add_argument("--aux_split", default="validation")
    return parser


def main(argv=None):
    process_baselines(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
