"""
scale_precision_ablation_se_to_ci.py

Read results/precision_ablation/summary_table.csv (produced by
experiment_precision_ablation.py), multiply the three SE columns by 1.96
to obtain 95% confidence interval half-widths, and write a new CSV.

SE columns scaled:
  se_acc          → ci95_acc
  se_ece_uwb_10   → ci95_ece_uwb_10
  se_brier_score  → ci95_brier_score

All other columns are carried over unchanged.

Usage:
    python3 src/calculating_standard_errors/scale_precision_ablation_se_to_ci.py
"""

import argparse
import csv
import os

SCALE = 1.96

SE_COLUMNS = ["se_acc", "se_ece_uwb_10", "se_brier_score"]

INPUT_PATH  = "./results/precision_ablation/summary_table.csv"
OUTPUT_PATH = "./results/precision_ablation/summary_table_ci95.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = reader.fieldnames
        rows = list(reader)

    # Build output fieldnames: insert ci95_* after each se_* column
    out_fields = []
    for col in original_fields:
        out_fields.append(col)
        if col in SE_COLUMNS:
            out_fields.append("ci95_" + col[3:])   # strip leading "se_"

    # Scale SE → CI and add ci95_* values
    out_rows = []
    for row in rows:
        new_row = {}
        for col in original_fields:
            new_row[col] = row[col]
            if col in SE_COLUMNS and row[col] not in ("", "nan"):
                try:
                    new_row["ci95_" + col[3:]] = f"{float(row[col]) * SCALE:.8f}"
                except ValueError:
                    new_row["ci95_" + col[3:]] = ""
            elif col in SE_COLUMNS:
                new_row["ci95_" + col[3:]] = row[col]   # preserve empty/nan as-is
        out_rows.append(new_row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Written {len(out_rows)} rows → {args.output}")
    print(f"Added CI columns: {['ci95_' + c[3:] for c in SE_COLUMNS]}")


if __name__ == "__main__":
    main()
