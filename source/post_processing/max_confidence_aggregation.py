import argparse
import json
import os
from source.post_processing.cli import add_prediction_arguments
from source.post_processing.records import parse_record, align_records
from source.post_processing.utils import load_data

# ==========================================
# Helper Functions
# ==========================================

def prepare_data(data1, data2):
    """Align predictions by ID while preserving primary record order."""
    return align_records(data1, data2, include_reference_correctness=True)


def process_max_confidence_aggregation(args):
    """
    For each data point, follow the prediction from the model with higher confidence.
    """
    # Load data
    confidence_pair_dict = load_data(args)
    
    # Construct Output Directory
    base_output_dir = os.path.join(args.output_dir, args.dataset_name, args.split, 
                                    'post_processing', 'max_confidence', 'confidence')
    
    if args.model1_type == 'LLMs' and args.model1_method and args.model1_name:
        model1_path_name = args.model1_name.replace('/', '_')
        sub_path = os.path.join(args.model1_method, model1_path_name)
    else:
        sub_path = "unknown_model1"

    if args.model2_name:
        model2_path_name = args.model2_name.replace('/', '_')
    else:
        model2_path_name = "unknown_model2"
        
    output_dir = os.path.join(base_output_dir, sub_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Iterate over file pairs
    for (name1, name2), (data1, data2) in confidence_pair_dict.items():
        if ('baseline' in name1 and 'baseline' in name2):
            if args.debug:
                data1 = data1[:5]
                data2 = data2[:5]
                print(f"Truncated to 5 records for debug mode")

            print(f"\n------ Processing ({name1}, {name2}) ------")
            
            # 1. Prepare aligned records
            records = prepare_data(data1, data2)
            if not records:
                print("No matching records found. Skipping.")
                continue
                
            print(f"Total aligned records: {len(records)}")
            
            # 2. Apply Max Confidence Aggregation
            processed_results = []
            m1_selected_count = 0
            m2_selected_count = 0
            
            for r in records:
                # Compare confidences and select the prediction with higher confidence
                if r['m1_conf'] >= r['m2_conf']:
                    # Follow model 1's prediction
                    selected_pred = r['m1_pred']
                    selected_conf = r['m1_conf']
                    m1_selected_count += 1
                else:
                    # Follow model 2's prediction
                    selected_pred = r['m2_pred']
                    selected_conf = r['m2_conf']
                    m2_selected_count += 1
                
                processed_results.append({
                    "id": r['id'],
                    "choices": r['choices'],
                    "gold": r['gold'],
                    "model_prediction": {
                        "final_answer": selected_pred,
                        "confidence": round(float(selected_conf), 4)
                    }
                })
            
            print(f"Model 1 selected: {m1_selected_count} times ({m1_selected_count/len(records)*100:.2f}%)")
            print(f"Model 2 selected: {m2_selected_count} times ({m2_selected_count/len(records)*100:.2f}%)")
            
            # 3. Save Results
            output_filepath = os.path.join(output_dir, f'{name1}_{model2_path_name}_{name2}.jsonl')
            with open(output_filepath, 'w', encoding='utf-8') as f:
                for item in processed_results:
                    f.write(json.dumps(item) + '\n')
                    
            print(f"Saved {len(processed_results)} processed records to {output_filepath}")


def build_parser():
    return add_prediction_arguments(argparse.ArgumentParser(
        description="Select the prediction with higher confidence; ties favor model 1."
    ))


def main(argv=None):
    process_max_confidence_aggregation(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
