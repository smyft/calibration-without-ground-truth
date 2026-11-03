"""Run an experiment as ``python -m src.experiments NAME [OPTIONS]``."""

import argparse
import importlib
import sys

EXPERIMENTS = {
    "main": "src.experiment_main",
    "all-baselines": "src.experiments.experiment_all_baselines",
    "label-free": "src.experiments.experiment_label_free_baselines",
    "dummy-reference": "src.experiments.experiment_dummy_reference",
    "precision": "src.experiments.experiment_precision_ablation",
    "equal-mass-ece": "src.experiments.experiment_equal_mass_ece",
    "failure-analysis": "src.experiments.experiment_failure_mode_analysis",
    "supervised-at-nstar": "src.experiments.experiment_supervised_baselines_at_nstar",
    "sample-complexity": "src.experiments.run_all_sample_complexity",
    "reference-validation": "src.experiments.sample_complexity_reference_validation",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        return
    selection = parser.parse_args(argv[:1])
    module = importlib.import_module(EXPERIMENTS[selection.experiment])
    module.main(argv[1:])


if __name__ == "__main__":
    main()
