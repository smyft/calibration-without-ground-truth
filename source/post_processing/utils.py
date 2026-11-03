import os
import json
import numpy as np
from itertools import product

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


def confidence2score(confidence, is_pos_class):
    """For a binary classification task, convert (confidence, is_pos_class) to probability of positive class"""
    if confidence < 0 or confidence > 1:
        return -1
    if is_pos_class > 0.5:
        return confidence
    else:
        return 1 - confidence


def get_confidence_dir(args, model_prefix, split=None, dataset_name=None):
    """
    Helper function to construct confidence directory path based on model type.
    
    Args:
        args: Arguments object
        model_prefix: 'model1' or 'model2'
        split: Optional split name. If None, uses args.split
        dataset_name: Optional dataset name. If None, uses args.dataset_name
    """
    if split is None:
        split = args.split
    if dataset_name is None:
        dataset_name = args.dataset_name
        
    model_type = getattr(args, f'{model_prefix}_type')
    
    if model_type == 'LLMs':
        method_name = getattr(args, f'{model_prefix}_method')
        model_name = getattr(args, f'{model_prefix}_name')
        if not method_name or not model_name:
            print(f"Warning: Method and model name should be specified for LLMs ({model_prefix}).")
            return None
        
        model_name_for_path = model_name.replace('/', '_')
        return os.path.join(args.confidence_dir, dataset_name, split, 'LLMs', 'confidence', method_name, model_name_for_path)
    
    elif model_type == 'mathematical_methods':
        model_name = getattr(args, f'{model_prefix}_name')
        if not model_name:
            print(f"Warning: Model name should be specified for mathematical methods ({model_prefix}).")
            return None
        
        model_name_for_path = model_name.replace('/', '_')
        return os.path.join(args.confidence_dir, dataset_name, split, 'mathematical_methods', 'confidence', model_name_for_path)
    
    else:
        print(f"Unexpected type: {model_type}")
        return None


def load_data(args, split=None, dataset_name=None):
    """
    Loads confidence data from two specified models.
    
    Args:
        args: Arguments object
        split: Optional split name. If None, uses args.split
        dataset_name: Optional dataset name. If None, uses args.dataset_name
    """
    # Construct paths
    dir1 = get_confidence_dir(args, 'model1', split=split, dataset_name=dataset_name)
    dir2 = get_confidence_dir(args, 'model2', split=split, dataset_name=dataset_name)

    if not dir1 or not os.path.exists(dir1):
        print(f"Directory not found for model1: {dir1}")
        return {}
    if not dir2 or not os.path.exists(dir2):
        print(f"Directory not found for model2: {dir2}")
        return {}
    
    files1 = [f for f in os.listdir(dir1) if f.endswith('.jsonl')]
    files2 = [f for f in os.listdir(dir2) if f.endswith('.jsonl')]
     
    result = dict()
    for file1, file2 in product(files1, files2):
        cleaned_file1 = os.path.splitext(file1)[0]
        cleaned_file2 = os.path.splitext(file2)[0]

        if ('baseline' in cleaned_file1 and 'baseline' in cleaned_file2):
            
            print(f"\n------ Loading Confidence Files: ({cleaned_file1}, {cleaned_file2}) ------")
            path1 = os.path.join(dir1, file1)
            print(f"Loading model1 confidence data from: {path1}")
            data1 = load_confidence_results(path1)

            path2 = os.path.join(dir2, file2)
            print(f"Loading model2 confidence data from: {path2}")
            data2 = load_confidence_results(path2)

            if not data1 or not data2:
                print("Warning! One or both confidence data files could not be loaded.")
                continue

            if len(data1) != len(data2):
                print("Warning! The number of records in the two confidence files is different.")
            
            result[(cleaned_file1, cleaned_file2)] = (data1, data2)

    return result


def load_application_data(args, split=None, dataset_name=None):
    """
    Loads confidence data from model1 only for application.
    Used when applying calibration map to a different split or dataset.
    
    Args:
        args: Arguments object
        split: Optional split name. If None, uses args.application_split if available
        dataset_name: Optional dataset name. If None, uses args.application_dataset_name if available
    """
    if split is None:
        split = getattr(args, 'application_split', args.split)
    if dataset_name is None:
        dataset_name = getattr(args, 'application_dataset_name', args.dataset_name)
    
    # Construct path for model1 only
    dir1 = get_confidence_dir(args, 'model1', split=split, dataset_name=dataset_name)

    if not dir1 or not os.path.exists(dir1):
        print(f"Directory not found for model1: {dir1}")
        return {}
    
    files1 = [f for f in os.listdir(dir1) if f.endswith('.jsonl')]
     
    result = dict()
    for file1 in files1:
        cleaned_file1 = os.path.splitext(file1)[0]

        if 'baseline' in cleaned_file1:
            
            print(f"\n------ Loading Application Data: {cleaned_file1} ------")
            path1 = os.path.join(dir1, file1)
            print(f"Loading model1 application data from: {path1}")
            data1 = load_confidence_results(path1)

            if not data1:
                print("Warning! Application data could not be loaded.")
                continue
            
            result[cleaned_file1] = data1

    return result


def calculate_joint_distribution_for_binary_prediction(confidence_data1, confidence_data2, debug=False):
    """
    Calculates the joint distribution of confidences from two different methods.
    """
    if not confidence_data1 or not confidence_data2:
        print("Confidence data is missing.")
        return None
    
    if len(confidence_data1) != len(confidence_data2):
        new_length = min(len(confidence_data1), len(confidence_data2))
        print(f"Warning! The number of records in the two confidence files is different. Truncating to shorter length {new_length}")
        confidence_data1 = confidence_data1[:new_length]
        confidence_data2 = confidence_data2[:new_length]

    if debug:
        confidence_data1 = confidence_data1[:5]
        confidence_data2 = confidence_data2[:5]

    score_pairs = list()
    records_data = list()

    for record_idx, (record1, record2) in enumerate(zip(confidence_data1, confidence_data2)):
        try:
            if 'id' not in record1 or 'id' not in record2:
                print(f"Skipping record {record_idx}: 'id' missing.")
                continue
            
            id1 = record1['id']
            id2 = record2['id']
            
            if id1 != id2:
                print(f"Skipping record {record_idx}: id mismatch between models.")
                continue
            
            pos_class = record1['choices'][0] # we assume the first class is positive class

            ans1 = record1['model_prediction']['final_answer']
            conf1 = record1['model_prediction']['confidence']
            is_pos_class1 = (ans1 == pos_class)
            score1 = confidence2score(conf1, is_pos_class1)
            score1 = round(score1, 4)
            
            ans2 = record2['model_prediction']['final_answer']
            conf2 = record2['model_prediction']['confidence']
            is_pos_class2 = (ans2 == pos_class)
            score2 = confidence2score(conf2, is_pos_class2)
            score2 = round(score2, 4)

            if score1 != -1 and score2 != -1:
                score_pairs.append((score1, score2))
                records_data.append({
                    'id': record1.get('id'),
                    'choices': record1.get('choices'),
                    'gold': record1.get('gold'),
                })
            else:
                print(f"Skipping record {record_idx}: prediction1=({ans1}, {conf1}), prediction1=({ans2}, {conf2})")

        except KeyError as e:
            print(f"Skipping record due to missing key: {e}")
            continue

    if not score_pairs:
        print("No valid pairs of confidences found to create a joint distribution.")
        return

    print(f"\nCalculating joint distribution with {len(score_pairs)} data points.")

    joint_data = np.array(score_pairs)
    F1_samples = joint_data[:, 0]
    F2_samples = joint_data[:, 1]
    F1_vals = np.unique(F1_samples)
    F2_vals = np.unique(F2_samples)
    I = len(F1_vals)
    J = len(F2_vals)

    F1_map = {val: i for i, val in enumerate(F1_vals)}
    F2_map = {val: j for j, val in enumerate(F2_vals)}
    P_f1f2 = np.zeros((I, J))
    for f1, f2 in joint_data:
        i = F1_map[f1]
        j = F2_map[f2]
        P_f1f2[i, j] += 1
    P_f1f2 /= len(score_pairs)
        
    results = {
        'P_f1f2': P_f1f2,
        'F1_vals': F1_vals,
        'F2_vals': F2_vals,
        'records': records_data,
        'score_pairs': score_pairs,
    }

    return results


def calculate_joint_distribution_for_confidence(confidence_data1, confidence_data2, debug=False, precision=2, bin_width=None):
    """
    Calculates the joint distribution of (j, m) reports and prepares inputs for solve_minimax.
    Mimics calculate_joint_distribution but for general K-state reports defined by (predicted_class, confidence).

    Args:
        confidence_data1: Confidence records for the primary (instruct) model.
        confidence_data2: Confidence records for the reference (base) model.
        debug: If True, truncate to 5 records for fast debugging.
        precision: Legacy input-grid decimal places (fallback when bin_width is None).
                   precision=2 → s=0.01 (100 bins), default used in paper experiments.
                   This does not set the final Ours confidence output precision.
        bin_width: Float step size for discretization. If provided, overrides precision.
                   E.g. bin_width=0.05 places confidences on {0, 0.05, 0.10, ..., 1.0}.
    """
    if not confidence_data1 or not confidence_data2:
        print("Confidence data is missing.")
        return None
    
    if len(confidence_data1) != len(confidence_data2):
        new_length = min(len(confidence_data1), len(confidence_data2))
        print(f"Warning! The number of records in the two confidence files is different. Truncating to shorter length {new_length}")
        confidence_data1 = confidence_data1[:new_length]
        confidence_data2 = confidence_data2[:new_length]

    if debug:
        confidence_data1 = confidence_data1[:5]
        confidence_data2 = confidence_data2[:5]

    # Resolve effective bin width for discretization
    _bin_width = bin_width if bin_width is not None else 10.0 ** (-precision)

    def _disc(x):
        """Round x to the nearest _bin_width using round-half-up (not banker's rounding).
        Python's built-in round() uses banker's rounding (round-half-to-even), which maps
        boundary values like 0.25 below 1/K when s=0.1 (0.25/0.1=2.5 rounds to 2 → 0.2).
        math.floor(x/s + 0.5) gives standard round-half-up behaviour.
        """
        import math
        return round(math.floor(x / _bin_width + 0.5) * _bin_width, 10)

    # Assume MMLU-style format where 'choices' is present in records
    # Derive K and mapping from the first record
    first_rec = confidence_data1[0]
    if 'choices' in first_rec and isinstance(first_rec['choices'], list):
        choices = first_rec['choices']
        K = len(choices)
        choice_map = {c: i for i, c in enumerate(choices)}
    else:
        # Fallback for binary or consistent with older formats if possible
        # Default to K=2 if undefined
        K = 2
        choice_map = None # Expect prediction to be integer 0 or 1

    records_data = list()
    
    # Store tuples of ((j1, m1), (j2, m2))
    raw_pairs = list()

    for record_idx, (record1, record2) in enumerate(zip(confidence_data1, confidence_data2)):
        try:
            if 'id' not in record1 or 'id' not in record2:
                print(f"Skipping record {record_idx}: 'id' missing.")
                continue
            
            id1 = record1['id']
            id2 = record2['id']
            
            if id1 != id2:
                print(f"Skipping record {record_idx}: id mismatch between models.")
                continue

            pred1 = record1['model_prediction']
            pred2 = record2['model_prediction']
            
            ans1 = pred1['final_answer']
            conf1 = pred1['confidence']
            
            ans2 = pred2['final_answer']
            conf2 = pred2['confidence']
            
            # Map answers to indices
            j1 = -1
            if choice_map and ans1 in choice_map:
                j1 = choice_map[ans1]
            elif isinstance(ans1, int):
                j1 = ans1
            elif isinstance(ans1, str) and ans1.isdigit():
                j1 = int(ans1)
            
            j2 = -1
            if choice_map and ans2 in choice_map:
                j2 = choice_map[ans2]
            elif isinstance(ans2, int):
                j2 = ans2
            elif isinstance(ans2, str) and ans2.isdigit():
                j2 = int(ans2)

            # Check validity
            if j1 < 0 or j1 >= K or j2 < 0 or j2 >= K:
                print(f"Skipping record {record_idx}: ans1={ans1}, ans2={ans2} not in valid range [0, {K})")
                continue
            
            # Reports — discretize to the effective bin width
            r1 = (j1, _disc(conf1))
            r2 = (j2, _disc(conf2))
            raw_pairs.append((r1, r2))

            records_data.append({
                'id': record1.get('id'),
                'choices': record1.get('choices'),
                'gold': record1.get('gold'),
            })

        except KeyError as e:
            print(f"Skipping record due to missing key: {e}")
            continue

    if not raw_pairs:
        print("No valid pairs found.")
        return None

    # Identify unique reports
    unique_r1 = sorted(list(set(p[0] for p in raw_pairs)))
    unique_r2 = sorted(list(set(p[1] for p in raw_pairs)))
    
    n_r1 = len(unique_r1)
    n_r2 = len(unique_r2)
    
    r1_map = {r: i for i, r in enumerate(unique_r1)}
    r2_map = {r: i for i, r in enumerate(unique_r2)}
    
    P_r1r2 = np.zeros((n_r1, n_r2))
    
    for r1, r2 in raw_pairs:
        idx1 = r1_map[r1]
        idx2 = r2_map[r2]
        P_r1r2[idx1, idx2] += 1
        
    P_r1r2 /= len(raw_pairs)
    
    # Extract vectors
    r1_j = np.array([r[0] for r in unique_r1])
    r1_m = np.array([r[1] for r in unique_r1])
    
    r2_j = np.array([r[0] for r in unique_r2])
    r2_m = np.array([r[1] for r in unique_r2])
    
    # Compute nnz from P_r1r2 (mirrors the solver's threshold)
    nnz = int(np.sum(P_r1r2 > 1e-12))

    # Also include score_pairs and records to match requested return style + extra
    results = {
        'P_r1r2': P_r1r2,
        'r1_j': r1_j,
        'r1_m': r1_m,
        'r2_j': r2_j,
        'r2_m': r2_m,
        'K': K,
        'records': records_data,
        'report_pairs': raw_pairs,
        # Grid statistics useful for sensitivity/ablation analysis
        'grid_stats': {
            'bin_width': _bin_width,
            'n_calib': len(raw_pairs),
            'n_r1': n_r1,
            'n_r2': n_r2,
            'total_cells': n_r1 * n_r2,
            'nnz': nnz,
            'sparsity': nnz / (n_r1 * n_r2) if n_r1 * n_r2 > 0 else 0.0,
            'fill_rate': nnz / len(raw_pairs) if raw_pairs else 0.0,
        },
    }

    if debug:
        gs = results['grid_stats']
        print(f"Constructed joint distribution with K={K}, shape P_r1r2={P_r1r2.shape}, "
              f"bin_width={_bin_width}, nnz={nnz}, sparsity={gs['sparsity']:.4f}")

    return results


def calculate_joint_distribution_for_mode(confidence_data1, confidence_data2, debug=False, precision=2):
    """
    Calculates the joint distribution of mode predictions (predicted class only) from two different methods.
    Similar to calculate_joint_distribution_for_confidence but only considers predicted class, not confidence.
    """
    if not confidence_data1 or not confidence_data2:
        print("Confidence data is missing.")
        return None
    
    if len(confidence_data1) != len(confidence_data2):
        new_length = min(len(confidence_data1), len(confidence_data2))
        print(f"Warning! The number of records in the two confidence files is different. Truncating to shorter length {new_length}")
        confidence_data1 = confidence_data1[:new_length]
        confidence_data2 = confidence_data2[:new_length]

    if debug:
        confidence_data1 = confidence_data1[:5]
        confidence_data2 = confidence_data2[:5]

    # Derive K from the first record
    first_rec = confidence_data1[0]
    if 'choices' in first_rec and isinstance(first_rec['choices'], list):
        choices = first_rec['choices']
        K = len(choices)
        choice_map = {c: i for i, c in enumerate(choices)}
    else:
        # Fallback
        K = 2
        choice_map = None

    records_data = list()
    
    # Store tuples of (r1, r2) where r1 and r2 are just the predicted class indices
    raw_pairs = list()

    for record_idx, (record1, record2) in enumerate(zip(confidence_data1, confidence_data2)):
        try:
            if 'id' not in record1 or 'id' not in record2:
                print(f"Skipping record {record_idx}: 'id' missing.")
                continue
            
            id1 = record1['id']
            id2 = record2['id']
            
            if id1 != id2:
                print(f"Skipping record {record_idx}: id mismatch between models.")
                continue

            pred1 = record1['model_prediction']
            pred2 = record2['model_prediction']
            
            ans1 = pred1['final_answer']
            ans2 = pred2['final_answer']
            
            # Map answers to indices
            j1 = -1
            if choice_map and ans1 in choice_map:
                j1 = choice_map[ans1]
            elif isinstance(ans1, int):
                j1 = ans1
            elif isinstance(ans1, str) and ans1.isdigit():
                j1 = int(ans1)
            
            j2 = -1
            if choice_map and ans2 in choice_map:
                j2 = choice_map[ans2]
            elif isinstance(ans2, int):
                j2 = ans2
            elif isinstance(ans2, str) and ans2.isdigit():
                j2 = int(ans2)

            # Check validity
            if j1 < 0 or j1 >= K or j2 < 0 or j2 >= K:
                print(f"Skipping record {record_idx}: ans1={ans1}, ans2={ans2} not in valid range [0, {K})")
                continue
            
            # For mode, we only care about the predicted class
            r1 = j1
            r2 = j2
            raw_pairs.append((r1, r2))

            records_data.append({
                'id': record1.get('id'),
                'choices': record1.get('choices'),
                'gold': record1.get('gold'),
                'model_prediction': record1.get('model_prediction'),  # Keep original prediction
            })

        except KeyError as e:
            print(f"Skipping record due to missing key: {e}")
            continue

    if not raw_pairs:
        print("No valid pairs found.")
        return None

    # Identify unique reports (just class indices now)
    unique_r1 = sorted(list(set(p[0] for p in raw_pairs)))
    unique_r2 = sorted(list(set(p[1] for p in raw_pairs)))
    
    n_r1 = len(unique_r1)
    n_r2 = len(unique_r2)
    
    r1_map = {r: i for i, r in enumerate(unique_r1)}
    r2_map = {r: i for i, r in enumerate(unique_r2)}
    
    P_r1r2 = np.zeros((n_r1, n_r2))
    
    for r1, r2 in raw_pairs:
        idx1 = r1_map[r1]
        idx2 = r2_map[r2]
        P_r1r2[idx1, idx2] += 1
        
    P_r1r2 /= len(raw_pairs)
    
    # Convert to arrays
    r1_vals = np.array(unique_r1)
    r2_vals = np.array(unique_r2)
    
    results = {
        'P_r1r2': P_r1r2,
        'r1_vals': r1_vals,
        'r2_vals': r2_vals,
        'K': K,
        'records': records_data,
        'report_pairs': raw_pairs
    }
    
    if debug:
        print(f"Constructed joint distribution for mode with K={K}, shape P_r1r2={P_r1r2.shape}")
        print(f"Unique r1 values: {r1_vals}")
        print(f"Unique r2 values: {r2_vals}")
    
    return results


def calculate_joint_distribution_for_full_prediction(confidence_data1, confidence_data2, debug=False, precision=2):
    """
    Calculates the joint distribution of full predictions (probability distributions) from two different methods.
    Based on logprobs_raw.
    """
    if not confidence_data1 or not confidence_data2:
        print("Confidence data is missing.")
        return None
    
    if len(confidence_data1) != len(confidence_data2):
        new_length = min(len(confidence_data1), len(confidence_data2))
        print(f"Warning! The number of records in the two confidence files is different. Truncating to shorter length {new_length}")
        confidence_data1 = confidence_data1[:new_length]
        confidence_data2 = confidence_data2[:new_length]

    if debug:
        confidence_data1 = confidence_data1[:5]
        confidence_data2 = confidence_data2[:5]

    records_data = list()
    full_prediction_pairs = list()

    for record_idx, (record1, record2) in enumerate(zip(confidence_data1, confidence_data2)):
        try:
            if 'id' not in record1 or 'id' not in record2:
                print(f"Skipping record {record_idx}: 'id' missing.")
                continue
            
            id1 = record1['id']
            id2 = record2['id']
            
            if id1 != id2:
                print(f"Skipping record {record_idx}: id mismatch between models.")
                continue

            # Check consistency of choices
            if 'choices' not in record1 or 'choices' not in record2:
                print(f"Skipping record {record_idx}: 'choices' missing.")
                continue
            
            choices1 = record1['choices']
            choices2 = record2['choices']
            
            if choices1 != choices2:
                print(f"Skipping record {record_idx}: choices mismatch between models.")
                continue

            # Process model 1
            if 'logprobs_raw' not in record1:
               print(f"Skipping record {record_idx}: 'logprobs_raw' missing in model 1.")
               continue
            
            logprobs1 = record1['logprobs_raw']
            # Ensure we extract probs in the order of choices
            vals1 = []
            for c in choices1:
                vals1.append(logprobs1.get(c, -np.inf)) # Default to very low logprob if missing
            
            # Convert to probabilities
            vals1 = np.array(vals1)
            # Softmax: exp(x - max(x)) / sum(exp(x - max(x)))
            e_x1 = np.exp(vals1 - np.max(vals1))
            probs1 = e_x1 / e_x1.sum()
            
            # Process model 2
            if 'logprobs_raw' not in record2:
               print(f"Skipping record {record_idx}: 'logprobs_raw' missing in model 2.")
               continue
            
            logprobs2 = record2['logprobs_raw']
            vals2 = []
            for c in choices2:
                vals2.append(logprobs2.get(c, -np.inf))
            
            vals2 = np.array(vals2)
            e_x2 = np.exp(vals2 - np.max(vals2))
            probs2 = e_x2 / e_x2.sum()

            probs1 = np.round(probs1, precision)
            probs2 = np.round(probs2, precision)

            full_prediction_pairs.append((probs1, probs2))

            records_data.append({
                'id': record1.get('id'),
                'choices': record1.get('choices'),
                'gold': record1.get('gold'),
            })

        except KeyError as e:
            print(f"Skipping record due to missing key: {e}")
            continue

    if not full_prediction_pairs:
        print("No valid pairs found.")
        return None

    # Calculate joint distribution P(q1, q2)
    # Map vector -> index
    unique_q1 = set()
    unique_q2 = set()
    
    for p1, p2 in full_prediction_pairs:
        unique_q1.add(tuple(p1))
        unique_q2.add(tuple(p2))
    
    sorted_q1 = sorted(list(unique_q1))
    sorted_q2 = sorted(list(unique_q2))
    
    q1_map = {q: i for i, q in enumerate(sorted_q1)}
    q2_map = {q: i for i, q in enumerate(sorted_q2)}
    
    N = len(sorted_q1)
    M = len(sorted_q2)
    
    P_q1q2 = np.zeros((N, M))
    
    for p1, p2 in full_prediction_pairs:
        i = q1_map[tuple(p1)]
        j = q2_map[tuple(p2)]
        P_q1q2[i, j] += 1
        
    P_q1q2 /= len(full_prediction_pairs)
    
    q1_values = np.array(sorted_q1)
    q2_values = np.array(sorted_q2)
    
    K = q1_values.shape[1] if N > 0 else 0

    results = {
        'P_q1q2': P_q1q2,
        'q1_values': q1_values,
        'q2_values': q2_values,
        'K': K,
        'records': records_data,
        'report_pairs': full_prediction_pairs
    }
    
    return results
