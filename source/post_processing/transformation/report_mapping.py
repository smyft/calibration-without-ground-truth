"""Pure deployment helpers for confidence-report calibration.

Callers quantize reports with the same grid used to construct calibration
support. Only an exact (choice, quantized confidence) key selects a map. On a
miss, callers retain the quantized report according to their existing record
filtering and absent-choice rules. No approximate lookup is performed.
"""

OUTPUT_PRECISION = 4
DEFAULT_BIN_WIDTH = 0.01
MAPPING_PROTOCOL = "quantized-exact-or-identity-v1"


def select_report_mapping(mapping, r1_j, r1_m, choice, confidence):
    """Return the exact map or None so the caller keeps the quantized report.

    ``confidence`` is already quantized. The support arrays are accepted for
    call compatibility; they never provide a substitute for a missing key.
    """
    return mapping.get((choice, confidence))


def output_confidence(confidence):
    """Use the main experiment's Python round convention and numeric output."""
    return round(float(confidence), OUTPUT_PRECISION)
