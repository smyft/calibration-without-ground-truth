from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import DATASETS, MODEL_IDS

import os
import json
import argparse
import numpy as np

def load_confidence_results(file_path: str):
    """Loads confidence results from a .jsonl file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def calculate_brier_score(is_correct, confidences):
    """
    Calculates the Brier Score.
    :param is_correct: A list or numpy array of binary outcomes (0 or 1).
    :param confidences: A list or numpy array of confidences.
    :return: The Brier Score.
    """
    is_correct = np.array(is_correct)
    confidences = np.array(confidences)
    return np.mean((confidences - is_correct)**2)

def calculate_ece(is_correct, confidences, n_bins=10, debug=False):
    """
    Calculates the Expected Calibration Error (ECE).
    :param is_correct: A list or numpy array of binary outcomes (0 or 1).
    :param confidences: A list or numpy array of confidence scores.
    :param n_bins: The number of bins to use for calibration.
    :return: The ECE value.
    """
    is_correct = np.array(is_correct)
    confidences = np.array(confidences)

    if debug:
        print("\n---- Calculating ECE ----")
        print(f"Is_correct: {is_correct}")
        print(f"Confidences: {confidences}")

    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_lower == bin_limits[0]:
            in_bin = (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if np.any(in_bin):
            if debug:
                print(f"\n    In Bin ({bin_lower}, {bin_upper})")
                print(f"    Is_correct: {is_correct[in_bin]}")
                print(f"    Confidences: {confidences[in_bin]}")
            bin_prob = np.mean(in_bin)
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            avg_accuracy_in_bin = np.mean(is_correct[in_bin])
            ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * bin_prob

    return ece

def calculate_proper_score(is_correct, confidences):
    """
    Calculates the Proper Score for Confidence.
    :param is_correct: A list or numpy array of binary outcomes (0 or 1).
    :param confidences: A list or numpy array of confidences.
    :return: The Proper Score.
    """
    is_correct = np.array(is_correct)
    confidences = np.array(confidences)
    return np.mean(is_correct - (confidences - is_correct)**2)

def calculate_scores(confidence_filepath, output_filepath, debug=False):
    print(f"\nLoading confidence data from: {confidence_filepath}")
    confidence_data = load_confidence_results(confidence_filepath)
    if not confidence_data:
        print("No confidence data loaded. Exiting.")
        return

    confidences = []
    predictions = []
    groundtruth = []
    is_correct = []

    for record_idx, record in enumerate(confidence_data):
        if debug:
            if record_idx >= 5:
                break
        confidences.append(record['model_prediction']['confidence'])
        predictions.append(record['model_prediction']['final_answer'])
        groundtruth.append(record['choices'][record['gold']])
        is_correct.append(record['choices'][record['gold']] == record['model_prediction']['final_answer'])

    if not confidences:
        print("No valid confidence scores found to calculate calibration scores.")
        return

    brier_score = calculate_brier_score(is_correct, confidences)
    ece_score = calculate_ece(is_correct, confidences, debug=debug)
    proper_score = calculate_proper_score(is_correct, confidences)
    accuracy = np.mean(is_correct)
    print(f"Brier Score: {brier_score:.4f}")
    print(f"ECE: {ece_score:.4f}")
    print(f"Proper Score: {proper_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    results_to_save = {
        'brier_score': round(brier_score, 4),
        'ece': round(ece_score, 4),
        'proper_score': round(proper_score, 4),
        'accuracy': round(accuracy, 4),
        'n_samples': len(confidences)
    }

    # Save the results
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=4)

    print(f"\nCalibration scores saved to {output_filepath}")

def main(args):
    # Determine the directory based on the method name
    if args.method_type == 'LLMs':
        if not args.llm_method_name or not args.llm_model_name:
            print(f"Warning: Method and model name should be specified for LLMs.")
            return

        llm_model_name_for_path = args.llm_model_name.replace('/', '_')
        confidence_dir = os.path.join(args.confidence_dir, args.dataset_name, args.split, 'LLMs', 'confidence', args.llm_method_name, llm_model_name_for_path)
    elif args.method_type == 'post_processing':
        if not args.post_processing_method or not args.llm_method_name or not args.llm_model_name:
            print(f"Warning: Method and model name should be specified for post processing methods.")
            return

        llm_model_name_for_path = args.llm_model_name.replace('/', '_')
        confidence_dir = os.path.join(args.confidence_dir, args.dataset_name, args.split, 'post_processing', args.post_processing_method)
        if args.post_processing_method == 'transformation':
            if args.post_processing_type:
                confidence_dir = os.path.join(confidence_dir, args.post_processing_type)
            calibration_dataset = getattr(args, 'calibration_dataset_name', None) or args.dataset_name
            calibration_split = getattr(args, 'calibration_split', None) or args.split
            confidence_dir = os.path.join(
                confidence_dir, f'calibrated_on_{calibration_dataset}_{calibration_split}')
        confidence_dir = os.path.join(confidence_dir, 'confidence', args.llm_method_name, llm_model_name_for_path)

    if not os.path.exists(confidence_dir):
        print(f"Warning! Directory not found: {confidence_dir}")
        return

    confidence_files = [f for f in os.listdir(confidence_dir) if f.endswith('.jsonl')]
    parts = confidence_dir.rpartition("confidence")
    output_dir = parts[0] + "score" + parts[2]

    if not confidence_files:
        print('Warning! Can not find any confidence files. Exiting.')
        return

    for filename in confidence_files:
        confidence_filepath = os.path.join(confidence_dir, filename)
        cleaned_filename = os.path.splitext(filename)[0]
        output_filepath = os.path.join(output_dir, f'{cleaned_filename}.json')

        calculate_scores(confidence_filepath, output_filepath, debug=args.debug)

def build_parser():
    parser = argparse.ArgumentParser(description="Calculate calibration metrics for predictions.")

    # ===== Dataset =====
    parser.add_argument('--dataset_name', type=str, default='MMLU_Redux',
                        choices=DATASETS,
                        help="Name of the dataset.")
    parser.add_argument('--split', type=str, default='test',
                    help="Split of the dataset.")

    # ===== Specifying Predictions Generated Previously =====
    parser.add_argument('--confidence_dir', type=str, default='./results',
                        help="Directory storing the confidence .jsonl files.")
    parser.add_argument('--method_type', type=str, default='LLMs',
                        choices=['LLMs', 'post_processing'],
                        help="The general method used for prediction.")

    # ===== For LLMs =====
    parser.add_argument('--llm_method_name', type=str, default='logprob',
                        choices=['logprob'],
                        help="The specific method used for LLM prediction.")
    parser.add_argument('--llm_model_name', type=str, default='mistralai/Ministral-3-8B-Instruct-2512', choices=MODEL_IDS,
                        help='Name of the LLM used for predictions.')

    # ===== For post_processing =====
    parser.add_argument('--post_processing_method', type=str, default='transformation',
                        choices=['transformation', 'histogram_binning', 'isotonic_regression', 'temperature_scaling', 'max_confidence'],
                        help="The post-processing method to apply.")
    parser.add_argument('--post_processing_type', type=str, default='confidence',
                        help="The post-processing method type.")
    parser.add_argument('--calibration_dataset_name', choices=DATASETS, default=None,
                        help="Transformation calibration dataset; defaults to --dataset_name.")
    parser.add_argument('--calibration_split', default=None,
                        help="Transformation calibration split; defaults to --split.")

    # ===== Output & Logs =====
    parser.add_argument('--debug', action='store_true',
                        help="Debug mode.")

    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())
