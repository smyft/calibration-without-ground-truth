"""CI export schema checks; no fitting or experiment artifacts are required."""

import copy
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from src.calculating_standard_errors.scale_se_to_ci import (
    LEGACY_CI_ONLY, RAW_AND_CI, SCALE, export_rows, main,
)


class UncertaintyExportTests(unittest.TestCase):
    def test_raw_se_preserved_and_halfwidth_matches_legacy_rounding(self):
        fields = ["method", "se_acc", "se_brier", "se_cl", "se_ece"]
        original = [{"method": "Ours", "se_acc": "0.123457", "se_brier": "0.000001",
                     "se_cl": "", "se_ece": "0.000000"}]
        before = copy.deepcopy(original)
        rows, out_fields = export_rows(original, fields, input_type="raw-se")
        self.assertEqual(original, before)
        for column in fields[1:]:
            self.assertEqual(rows[0][column], original[0][column])
            expected = f"{float(original[0][column]) * SCALE:.6f}" if original[0][column] else ""
            self.assertEqual(rows[0]["ci95_halfwidth_" + column[3:]], expected)
        self.assertEqual(rows[0]["uncertainty_type"], RAW_AND_CI)
        repeated, repeated_fields = export_rows(rows, out_fields, input_type="raw-se")
        self.assertEqual((repeated, repeated_fields), (rows, out_fields))

    def test_legacy_halfwidth_never_rescaled_or_fabricated_as_raw(self):
        rows, fields = export_rows([{"se_acc": "0.024198"}], ["se_acc"],
                                   input_type="legacy-ci95-halfwidth")
        self.assertEqual(rows[0]["ci95_halfwidth_acc"], "0.024198")
        self.assertEqual(rows[0]["se_acc"], "")
        self.assertEqual(rows[0]["uncertainty_type"], LEGACY_CI_ONLY)
        with self.assertRaisesRegex(ValueError, "refusing repeated scaling"):
            export_rows(rows, fields, input_type="raw-se")
        with self.assertRaisesRegex(ValueError, "unmarked historical schema"):
            export_rows(rows, fields, input_type="legacy-ci95-halfwidth")

    def test_inconsistent_or_untyped_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            export_rows([{"se_acc": "0.1"}], ["se_acc"], input_type="guess")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            export_rows([{"se_acc": "0.1", "ci95_halfwidth_acc": "0.38416"}],
                        ["se_acc", "ci95_halfwidth_acc"], input_type="raw-se")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main([])

    def test_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.csv"
            output = Path(directory) / "ci.csv"
            source.write_text("method,se_acc\nOurs,0.123457\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--input", str(source), "--output", str(output), "--input-type", "raw-se"])
            with output.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["se_acc"], "0.123457")
            self.assertEqual(row["ci95_halfwidth_acc"], "0.241976")


if __name__ == "__main__":
    unittest.main()
