import numpy as np

def aggregation_for_binary_prediction(data, debug=False):
    if not data:
        print("Could not retrieve joint distributions. Exiting.")
        return None
    
    P_f1f2 = data['P_f1f2']
    F1_vals = data['F1_vals']
    F2_vals = data['F2_vals']
    records = data['records']
    score_pairs = data['score_pairs']

    # Calculate mu1 for each f2
    mu1_map = {}
    for j, p2 in enumerate(F2_vals):
        P_f2 = np.sum(P_f1f2[:, j])
        if P_f2 > 0:
            P_f1_given_f2 = P_f1f2[:, j] / P_f2
            mu1_map[p2] = np.sum(F1_vals * P_f1_given_f2)

    updated_records = []
    for record_idx, (record, score_pair) in enumerate(zip(records, score_pairs)):
        f1 = score_pair[0]
        f2 = score_pair[1]

        f2_idx = np.argmin(np.abs(F2_vals - f2))
        closest_f2 = F2_vals[f2_idx]

        if abs(closest_f2 - f2) < 1e-9 and closest_f2 in mu1_map:
            mu1 = mu1_map[closest_f2]
            new_score = f1 - mu1 + closest_f2
            # For out of range cases, simply clip should work
            new_score = np.clip(new_score, 0, 1)
            new_score = round(new_score, 4)

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
            print(f"Warning: F2 value {f2} not found in F2_vals for record {record_idx}. Skipping.")
            
    return updated_records
