"""Export named 95% CI half-widths without replacing raw standard errors.

The input type is explicit because historical CI files used ``se_*`` columns
for already-scaled half-widths. Those rounded files cannot recover the exact raw
SE: importing one leaves the raw columns blank and records that limitation.

Usage:
    python -m src.calculating_standard_errors.scale_se_to_ci \
        --input results/standard_errors/standard_errors.csv \
        --input-type raw-se \
        --output results/standard_errors/standard_errors_ci95.csv
"""

import argparse
import csv
import os

SE_COLUMNS = ["se_acc", "se_brier", "se_cl", "se_ece"]
SCALE = 1.96
INPUT_TYPES = ("raw-se", "legacy-ci95-halfwidth")
RAW_AND_CI = "raw_se_and_ci95_halfwidth"
LEGACY_CI_ONLY = "ci95_halfwidth_only_raw_se_unavailable"


def export_rows(rows, fieldnames, *, input_type):
    """Return copied rows and fields; retain the legacy six-decimal CI rounding."""
    if input_type not in INPUT_TYPES:
        raise ValueError(f"input_type must be one of {INPUT_TYPES}")
    if not fieldnames:
        raise ValueError("Input CSV has no header")
    columns = [column for column in SE_COLUMNS if column in fieldnames]
    if not columns:
        raise ValueError("Input CSV contains no recognized SE columns")
    output_fields = list(fieldnames)
    for column in columns:
        target = "ci95_halfwidth_" + column.removeprefix("se_")
        if target not in output_fields:
            output_fields.append(target)
    if "uncertainty_type" not in output_fields:
        output_fields.append("uncertainty_type")

    output_rows = []
    for source in rows:
        row = dict(source)
        declared = row.get("uncertainty_type", "")
        if input_type == "raw-se":
            if declared not in ("", RAW_AND_CI):
                raise ValueError(f"Input declares {declared!r}, not raw SE; refusing repeated scaling")
            for column in columns:
                target = "ci95_halfwidth_" + column.removeprefix("se_")
                value = row.get(column, "")
                halfwidth = f"{float(value) * SCALE:.6f}" if value != "" else ""
                existing = row.get(target, "")
                if existing != "" and (halfwidth == "" or float(existing) != float(halfwidth)):
                    raise ValueError(f"Existing {target} disagrees with raw {column}; refusing overwrite")
                row[target] = halfwidth
            row["uncertainty_type"] = RAW_AND_CI
        else:
            # The old exporter overwrote se_* in place. Do not divide the rounded
            # half-widths to fabricate raw SE, and never multiply them a second time.
            if declared or any("ci95_halfwidth_" + col.removeprefix("se_") in fieldnames
                               for col in columns):
                raise ValueError("legacy-ci95-halfwidth requires the unmarked historical schema")
            for column in columns:
                target = "ci95_halfwidth_" + column.removeprefix("se_")
                row[target] = row.get(column, "")
                row[column] = ""
            row["uncertainty_type"] = LEGACY_CI_ONLY
        output_rows.append(row)
    return output_rows, output_fields


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preserve raw SE and export named 95% CI half-widths.")
    parser.add_argument("--input", default="results/standard_errors/standard_errors.csv")
    parser.add_argument("--output", default="results/standard_errors/standard_errors_ci95.csv")
    parser.add_argument("--input-type", choices=INPUT_TYPES, required=True,
                        help="Declare raw SE or historical half-widths mislabeled as se_*; never inferred from filenames.")
    args = parser.parse_args(argv)

    with open(args.input, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows, fieldnames = export_rows(list(reader), reader.fieldnames, input_type=args.input_type)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written {len(rows)} rows → {args.output}")
    print(f"Input type: {args.input_type}; CI half-width = {SCALE} × raw SE (six decimals).")


if __name__ == "__main__":
    main()
