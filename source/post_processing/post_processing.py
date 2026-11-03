"""Apply a calibration map to saved model predictions.

Run from the code directory with
``python -m source.post_processing.post_processing --help``.
"""

import argparse
import json
import os

from source.post_processing.cli import add_prediction_arguments
from source.post_processing.utils import (
    load_data, load_application_data,
    calculate_joint_distribution_for_confidence,
    calculate_joint_distribution_for_full_prediction,
    calculate_joint_distribution_for_mode,
)
from source.post_processing.transformation.transformation import (
    transformation_for_confidence,
    transformation_for_full_prediction,
    transformation_for_mode,
)
from src.config import DATASET_CONFIG
from source.post_processing.transformation.report_mapping import OUTPUT_PRECISION


TRANSFORMATIONS = {
    "confidence": (calculate_joint_distribution_for_confidence, transformation_for_confidence),
    "full_prediction": (calculate_joint_distribution_for_full_prediction, transformation_for_full_prediction),
    "mode": (calculate_joint_distribution_for_mode, transformation_for_mode),
}


def post_processing(args):
    """Fit and apply the selected transformation, retaining saved JSONL layout."""
    if args.post_processing_method != "transformation":
        raise ValueError("Only transformation has an implemented post-processing pipeline.")
    if args.post_processing_type not in TRANSFORMATIONS:
        raise ValueError(f"Unknown post_processing_type: {args.post_processing_type}")
    calc_func, processing_func = TRANSFORMATIONS[args.post_processing_type]
    calibration_dataset = args.dataset_name
    calibration_split = getattr(args, "calibration_split", None) or args.split
    application_dataset = getattr(args, "application_dataset_name", None) or args.dataset_name
    application_split = getattr(args, "application_split", None) or args.split
    transfer = (calibration_dataset, calibration_split) != (application_dataset, application_split)
    print(f"Calibration: {calibration_dataset}/{calibration_split}")
    print(f"Application: {application_dataset}/{application_split}")

    confidence_pairs = load_data(args, split=calibration_split, dataset_name=calibration_dataset)
    application_records = (
        load_application_data(args, split=application_split, dataset_name=application_dataset)
        if transfer else {}
    )
    primary_path = (
        os.path.join(args.model1_method, args.model1_name.replace("/", "_"))
        if args.model1_type == "LLMs" and args.model1_method and args.model1_name
        else "unknown_model1"
    )
    reference_path = args.model2_name.replace("/", "_") if args.model2_name else "unknown_model2"
    output_dir = os.path.join(
        args.output_dir, application_dataset, application_split, "post_processing",
        args.post_processing_method, args.post_processing_type,
        f"calibrated_on_{calibration_dataset}_{calibration_split}", "confidence", primary_path,
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    precision = args.precision
    if precision is None:
        precision = 2 if args.post_processing_type == "confidence" else 4
    distribution_args = {"debug": args.debug, "precision": precision}
    if args.post_processing_type == "confidence":
        # --precision remains only a legacy input-grid shorthand for confidence.
        bin_width = getattr(args, "bin_width", None)
        distribution_args["bin_width"] = (
            bin_width if bin_width is not None else 10.0 ** (-precision)
        )
    elif getattr(args, "bin_width", None) is not None:
        raise ValueError("--bin_width applies only to the confidence transformation.")

    for (name1, name2), (data1, data2) in confidence_pairs.items():
        joint_distribution = calc_func(data1, data2, **distribution_args)
        if not joint_distribution:
            print("Failed to calculate joint distribution. Skipping.")
            continue
        application_data = None
        if transfer:
            if name1 not in application_records:
                print(f"Application data not found for {name1}. Skipping.")
                continue
            records = application_records[name1]
            application_data = calc_func(records, records, **distribution_args)
            if not application_data:
                print(f"Failed to process application data for {name1}. Skipping.")
                continue

        output_path = os.path.join(output_dir, f"{name1}_{reference_path}_{name2}.jsonl")
        processing_args = {}
        if args.post_processing_type == "confidence":
            processing_args["diagnostics_path"] = output_path + ".diagnostics.json"
        result = processing_func(
            joint_distribution, debug=args.debug,
            precision=OUTPUT_PRECISION if args.post_processing_type == "confidence" else precision,
            application_data=application_data,
            **processing_args,
        )
        updated_data = result[0] if isinstance(result, tuple) else result
        if updated_data is None:
            print("Could not get updated results. Skipping.")
            continue
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in updated_data:
                handle.write(json.dumps(record) + "\n")
        print(f"Updated records saved to {output_path}")


def build_parser():
    parser = add_prediction_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--calibration_split", default=None)
    parser.add_argument("--application_dataset_name", choices=tuple(DATASET_CONFIG), default=None)
    parser.add_argument("--application_split", default=None)
    parser.add_argument("--post_processing_method", choices=("transformation",), default="transformation")
    parser.add_argument("--post_processing_type", choices=tuple(TRANSFORMATIONS), default="confidence")
    parser.add_argument("--precision", type=int, default=None,
                        help="For confidence: legacy input grid 10**(-precision), default 0.01; "
                             "output is always four decimal places. For other types: rounding precision (default 4).")
    parser.add_argument("--bin_width", type=float, default=None,
                        help="Confidence discretization width; overrides --precision for binning.")
    return parser


def main(argv=None):
    post_processing(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
