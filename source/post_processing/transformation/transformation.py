import numpy as np
from source.post_processing.transformation.binary_prediction import solve_calibration_qp
from source.post_processing.transformation.confidence import solve_minimax_sparse_vectorized
from source.post_processing.transformation.full_prediction import solve_optimal_transformation
from source.post_processing.transformation.mode import optimal_mode_mapping
from source.post_processing.transformation.report_mapping import output_confidence, select_report_mapping

def transformation_for_binary_prediction(data, debug=False, precision=4):
    """
    Applies a transformation based on a F1 mapping derived from optimization problem.
    """
    if not data:
        print("Could not retrieve joint distributions. Exiting.")
        return None

    P_f1f2 = data['P_f1f2']
    F1_vals = data['F1_vals']
    F2_vals = data['F2_vals']
    records = data['records']
    score_pairs = data['score_pairs']

    # Step 1: Solve the QP to get the calibrated F1 map
    print("Solving QP...")
    _, f1_map_values = solve_calibration_qp(P_f1f2, F1_vals, F2_vals, debug=debug)

    if f1_map_values is None:
        print(f"Failed to solve QP. Exiting.")
        return None
        
    # Create a dictionary for quick lookup: F1_val -> calibrated_F1_val
    f1_map = dict(zip(F1_vals, f1_map_values))

    print("Mapping model predictions...")

    updated_records = []
    for record_idx, (record, score_pair) in enumerate(zip(records, score_pairs)):
        f1 = score_pair[0]
        # Find the index of the current f1 in the F1_vals array
        # Handle potential floating point inaccuracies by finding the closest value
        f1_idx = np.argmin(np.abs(F1_vals - f1))
        closest_f1 = F1_vals[f1_idx]

        if abs(closest_f1 - f1) < 1e-9 and closest_f1 in f1_map:
            new_score = f1_map[closest_f1]
            # here solving minimax procedure should ensure that new_confidence in [0,1]
            new_score = np.clip(new_score, 0, 1)
            new_score = round(new_score, precision)

            new_record = record.copy()
            if new_score < 0.5:
                new_record['model_prediction'] = {
                    'final_answer': record['choices'][1],
                    'confidence': 1 - new_score,
                }
            else:
                new_record['model_prediction'] = {
                    'final_answer': record['choices'][0],
                    'confidence': new_score,
                }

            updated_records.append(new_record)
        else:
            print(f"Warning: F1 value {f1} not found in f1_map for record {record_idx}. Skipping.")

    return updated_records


def transformation_for_confidence(data, debug=False, precision=4, application_data=None,
                                  diagnostics_path=None):
    """
    Apply the fitted map to exact quantized reports; retain quantized misses.

    Args:
        data: Calibration data containing joint distribution (output of
              calculate_joint_distribution_for_confidence).
        debug: Debug mode flag.
        precision: Legacy compatibility argument. Output always uses four places,
                   independently of the bin_width used for discretization.
        application_data: Optional separate application data. If None, uses calibration data.
        diagnostics_path: Optional sidecar; does not change solve or failure handling.

    Returns:
        (updated_records, solver_stats): updated_records is the list of calibrated predictions;
        solver_stats is a dict with keys n_r1, n_r2, nnz, n_vars, solver_time_s, solver_status.
    """
    if not data:
        print("Could not retrieve joint distributions. Exiting.")
        return None

    P_r1r2 = data['P_r1r2']
    r1_j_vals = data['r1_j']
    r1_m_vals = data['r1_m']
    r2_j_vals = data['r2_j']
    r2_m_vals = data['r2_m']
    K = data['K']
    
    # Use application_data if provided, otherwise use calibration data
    if application_data is not None:
        records = application_data['records']
        report_pairs = application_data['report_pairs']
        print(f"Applying calibration map to {len(records)} application records...")
    else:
        records = data['records']
        report_pairs = data['report_pairs']
        print(f"Applying calibration map to {len(records)} calibration records...")

    # Step 1: Solve the QP to get the calibrated F1 map
    print("Solving minimax optimization problem...")
    diagnostic_args = {"diagnostics_path": diagnostics_path} if diagnostics_path is not None else {}
    solver_stats, r1_map_values = solve_minimax_sparse_vectorized(
        P_r1r2, r1_j_vals, r1_m_vals, r2_j_vals, r2_m_vals, K, debug=debug,
        **diagnostic_args,
    )
    if solver_stats:
        print(f"  Grid: {solver_stats['n_r1']}x{solver_stats['n_r2']}, "
              f"NNZ: {solver_stats['nnz']}, Vars: {solver_stats['n_vars']}, "
              f"Time: {solver_stats['solver_time_s']:.2f}s, "
              f"Status: {solver_stats['solver_status']}")

    if r1_map_values is None:
        print(f"Failed to solve minimax optimization problem. Exiting.")
        return None, solver_stats
    
    # Create a dictionary for quick lookup: F1_val -> calibrated_F1_val
    r1_map = dict()
    for i in range(len(r1_map_values)):
        r1_map[(r1_j_vals[i], r1_m_vals[i])] = r1_map_values[i]

    if debug:
        print("------")
        print("Recalibration Mapping:")
        print(r1_map)
        print("------")

    print("Mapping model predictions...")

    updated_records = []
    for record_idx, (record, report_pair) in enumerate(zip(records, report_pairs)):
        r1_j = report_pair[0][0]
        r1_m = report_pair[0][1]

        if not np.any(r1_j_vals == r1_j):
            print(f"Warning: r1 value ({r1_j}, {r1_m}) not found in r1_map for record {record_idx}. Skipping.")
            continue
        mapped = select_report_mapping(r1_map, r1_j_vals, r1_m_vals, r1_j, r1_m)
        if mapped is not None:
            new_r1_j, new_r1_m = mapped

            # here solving minimax procedure should ensure that new_confidence in [0,1]
            new_r1_m = np.clip(new_r1_m, 0, 1)
            new_r1_m = output_confidence(new_r1_m)

            new_record = record.copy()
            new_record['model_prediction'] = {
                'final_answer': record['choices'][new_r1_j],
                'confidence': new_r1_m
            }
            updated_records.append(new_record)
        else:
            print(f"Warning: r1 value ({r1_j}, {r1_m}) not found in r1_map for record {record_idx}. Keeping the quantized report.")
            new_record = record.copy()
            new_record['model_prediction'] = {
                'final_answer': record['choices'][r1_j],
                'confidence': output_confidence(r1_m)
            }
            updated_records.append(new_record)

    return updated_records, solver_stats


def transformation_for_mode(data, debug=False, precision=4, application_data=None):
    """
    Applies a transformation based on optimal mode mapping derived from LP optimization.
    Only considers the predicted class (mode), not confidence values.
    
    Args:
        data: Calibration data containing joint distribution
        debug: Debug mode flag
        precision: Rounding precision
        application_data: Optional separate application data. If None, uses calibration data.
    """
    if not data:
        print("Could not retrieve joint distributions. Exiting.")
        return None

    P_r1r2 = data['P_r1r2']
    r1_vals = data['r1_vals']
    r2_vals = data['r2_vals']
    K = data['K']
    
    # Use application_data if provided, otherwise use calibration data
    if application_data is not None:
        records = application_data['records']
        report_pairs = application_data['report_pairs']
        print(f"Applying calibration map to {len(records)} application records...")
    else:
        records = data['records']
        report_pairs = data['report_pairs']
        print(f"Applying calibration map to {len(records)} calibration records...")

    # Step 1: Solve the LP to get the optimal mode mapping
    print("Solving optimal mode mapping LP...")
    obj_val, f1_mapping = optimal_mode_mapping(P_r1r2, r1_vals, r2_vals, K, debug=debug)

    if f1_mapping is None or len(f1_mapping) == 0:
        print(f"Failed to solve optimal mode mapping. Exiting.")
        return None
        
    # Create a dictionary for quick lookup: r1_val -> new_class
    r1_map = dict()
    for i in range(len(r1_vals)):
        r1_map[r1_vals[i]] = f1_mapping[i]

    if debug:
        print(f"Mode mapping created: {r1_map}")

    print("Mapping model predictions...")

    updated_records = []
    for record_idx, (record, report_pair) in enumerate(zip(records, report_pairs)):
        r1 = report_pair[0]  # This is just the class index for mode

        if r1 not in r1_map:
            print(f"Warning: r1 value {r1} not found in r1_map for record {record_idx}. Skipping.")
            continue

        new_class = r1_map[r1]

        new_record = record.copy()
        
        # Update the prediction to the new class
        # Keep the original confidence if it exists
        if 'model_prediction' in record and 'confidence' in record['model_prediction']:
            original_confidence = record['model_prediction']['confidence']
        else:
            # If no confidence exists, we can set a default or keep None
            original_confidence = None
        
        # Map the new class index to the actual choice text
        if new_class < len(record['choices']):
            new_answer = record['choices'][new_class]
        else:
            print(f"Warning: new_class {new_class} out of range for choices in record {record_idx}. Skipping.")
            continue
        
        new_record['model_prediction'] = {
            'final_answer': new_answer
        }
        
        # Add confidence if it was present
        if original_confidence is not None:
            new_record['model_prediction']['confidence'] = original_confidence
        
        updated_records.append(new_record)

    return updated_records


def transformation_for_full_prediction(data, debug=False, precision=4, application_data=None):
    """
    Applies transformation for full prediction vectors.
    
    Args:
        data: Calibration data containing joint distribution
        debug: Debug mode flag
        precision: Rounding precision
        application_data: Optional separate application data. If None, uses calibration data.
    """
    if not data:
        print("Could not retrieve joint distributions. Exiting.")
        return None

    # Unpack data
    P_q1q2 = data['P_q1q2']
    q1_values = data['q1_values']
    q2_values = data['q2_values']
    K = data['K']
    
    # Use application_data if provided, otherwise use calibration data
    if application_data is not None:
        records = application_data['records']
        report_pairs = application_data['report_pairs']
        print(f"Applying calibration map to {len(records)} application records...")
    else:
        records = data['records']
        report_pairs = data['report_pairs']
        print(f"Applying calibration map to {len(records)} calibration records...")

    # Step 1: Solve to get f1 map
    print("Solving optimal transformation...")
    result_map = solve_optimal_transformation(P_q1q2, q1_values, q2_values, K, debug=debug)
    
    if not result_map:
        print("Failed to solve QP. Exiting.")
        return None

    # Step 2: Map predictions
    print("Mapping model predictions...")
    
    updated_records = []
    
    vector_lookup = {}
    for i, vec in enumerate(q1_values):
        vector_lookup[tuple(vec)] = result_map.get(i)

    for record_idx, (record, pair) in enumerate(zip(records, report_pairs)):
        probs1 = pair[0] # These are already rounded np.arrays
        
        vec_key = tuple(probs1)
        
        new_probs = None
        if vec_key in vector_lookup:
            new_probs = vector_lookup[vec_key]
        else:
            # Look for closest match if direct hit fails (should not happen often if consistent)
            diffs = q1_values - probs1
            dists = np.sum(diffs**2, axis=1)
            best_idx = np.argmin(dists)
            if dists[best_idx] < 1e-8:
                new_probs = result_map.get(best_idx)
            else:
                print(f"Warning: No matching q1 vector found for record {record_idx}. Distance: {dists[best_idx]}")
                continue
        
        if new_probs is None:
            continue

        # Extract new final answer and confidence
        best_idx = np.argmax(new_probs)
        new_conf = new_probs[best_idx]
        
        # Ensure index within range of choices
        if best_idx < len(record['choices']):
            final_ans = record['choices'][best_idx]
        else:
            # Fallback if inconsistent dimensions (shouldn't happen if K is consistent)
            final_ans = record['model_prediction']['final_answer'] 
        
        new_record = record.copy()
        new_record['model_prediction'] = {
            'final_answer': final_ans,
            'confidence': round(float(new_conf), precision)
        }
        new_record['full_prediction'] = np.round(new_probs, precision).tolist()
        updated_records.append(new_record)
        
    return updated_records
