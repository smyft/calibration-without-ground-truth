"""Shared dataset/model selection for experiment command-line entry points."""

from src.config import DATASETS, MODEL_PAIRS, MODEL_SPECS

FAMILIES = tuple(dict.fromkeys(spec.family for spec in MODEL_SPECS.values()))
SLUG_FAMILY = {spec.slug: spec.family for spec in MODEL_SPECS.values()}


def add_selection_arguments(parser, *, datasets=DATASETS):
    if "--datasets" not in parser._option_string_actions:
        parser.add_argument("--datasets", nargs="+", choices=datasets, default=None,
                            help="Evaluate only these datasets (default: experiment protocol).")
    parser.add_argument("--models", nargs="+", choices=FAMILIES, default=None,
                        help="Evaluate only these model pairs: qwen, llama, ministral.")


def selected_pairs(pairs, args):
    return [pair for pair in pairs
            if (not args.datasets or pair[0] in args.datasets)
            and (not args.models or SLUG_FAMILY[pair[1]] in args.models)]


def selected_models(args):
    return [(inst, base) for inst, base in MODEL_PAIRS
            if not args.models or SLUG_FAMILY[inst] in args.models]
