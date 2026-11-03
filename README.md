# Calibration without ground truth

This code generates multiple-choice model predictions, fits calibration maps
without labels, and evaluates calibration on held-out data. Run all commands
from the repository root, the directory containing this README and `src/`.

## Environment and installation

The validated environment uses Python **3.11.15** on Ubuntu **22.04.5 LTS**
(Linux x86-64). GPU inference was run with an NVIDIA RTX 6000D (about 84 GiB
VRAM), driver **595.71.05**, PyTorch **2.11.0** built for CUDA **13.0**,
vLLM **0.26.0**, Transformers **5.13.0**, and mistral-common **1.11.7**.
These describe the tested configuration; minimum GPU memory and compatibility
with other hardware or package versions have not been established. Context
length, batch size, and model configuration affect memory requirements.

`requirements.txt` pins the direct dependencies for the full workflow.
`requirements-analysis.txt` contains the shared dependencies needed to process
existing predictions, including PyTorch for the CPU PBR baseline. No API service
or API key is required. The dependency files are not a complete transitive lock;
record the resolved environment and run the tests when creating a new one.

For local-model inference and subsequent analysis, create a fresh Python 3.11
environment on a compatible Linux/CUDA machine:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Confirm that the installed PyTorch/vLLM CUDA builds are compatible with the
NVIDIA driver before inference. For CPU analysis of existing predictions,
use a separate environment and install the CPU PyTorch wheel first:

```bash
python3.11 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-analysis.txt
python -m pip check
python -B -m unittest discover -s tests -v
```

The CPU recipe omits vLLM and the model inference dependencies; it cannot
generate new LLM predictions. Calibration, baseline fitting, N* estimation,
and metric/uncertainty calculations run on CPU. PBR fitting and nested
resampling can still take substantial time. CPU computations were validated
in the full environment above; a fresh CPU-only installation must pass the
tests before use. No downloaded models or datasets are needed by the tests.

For a headless machine, set `MPLBACKEND=Agg`. Parallel CPU jobs may benefit
from explicit `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS`
limits to prevent oversubscription; record any chosen limits with the run.

## Supported models and data

`src/config.py` is the shared registry for these three base/instruct pairs:

| Model identifier | Local checkpoint directory under `MODEL_ROOT` |
| --- | --- |
| `Qwen/Qwen3-8B-Base` | `Qwen3-8B-Base` |
| `Qwen/Qwen3-8B` | `Qwen3-8B` |
| `meta-llama/Llama-3.1-8B` | `Llama-3.1-8B` |
| `meta-llama/Llama-3.1-8B-Instruct` | `Llama-3.1-8B-Instruct` |
| `mistralai/Ministral-3-8B-Base-2512` | `Ministral-3-8B-Base-2512` |
| `mistralai/Ministral-3-8B-Instruct-2512` | `Ministral-3-8B-Instruct-2512-BF16` |

Model loading always uses a local path. Missing configurations, tokenizers, or
weight shards cause an error before inference; the code never downloads model
weights. Override the checkpoint root with `--model_root` or `MODEL_ROOT`, or
use `--model_path` for an explicit local checkpoint. Precedence is
`--model_path`, then `--model_root`, then `MODEL_ROOT`, then the default
`CODE_ROOT.parent.parent / "models"`. Result names still use the canonical
identifier with `/` replaced by `_`, regardless of the local path.

Obtain the checkpoints separately using the model provider's download tools
and access permissions. Preserve the original configuration, tokenizer,
`safetensors` weights and index, and (for Ministral) `params.json` and
`tekken.json`. Record the exact upstream revision used; model names alone do
not identify immutable weights. Store any download credentials in the provider's
external credential store or environment, never in source files or command URLs.

For example, put downloads under the ignored `models/` and `datasets/`
directories and export their paths before running the commands below:

```bash
export MODEL_ROOT="$PWD/models"
export DATASET_ROOT="$PWD/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

These variables configure local input lookup; they do not download anything.
They may also point to storage outside the repository. `.env` files are ignored
by Git but are not loaded automatically; export variables in the shell or source
your own trusted local shell configuration.

| Dataset CLI name | Calibration data | Held-out evaluation data |
| --- | --- | --- |
| `MMLU` | MMLU validation | MMLU test |
| `MMLU_Redux` (MMLU-Redux) | MMLU validation | MMLU-Redux test |
| `CommonSenseQA` | CommonSenseQA train | CommonSenseQA validation |

The dataset loader reads standalone Arrow packages with `datasets.load_from_disk`.
Its default root is `CODE_ROOT.parent.parent / "datasets"`; set `DATASET_ROOT`
to override that root. Both inference inputs and few-shot examples use these
local paths:

```text
<DATASET_ROOT>/MMLU/prepared/<all-or-subject>/<split>/
<DATASET_ROOT>/MMLU-Redux-2.0/prepared/<subject>/test/
<DATASET_ROOT>/CommonSenseQA/prepared/default/<split>/
```

Each split must contain `state.json`, `dataset_info.json`, and the Arrow shards
listed in its state file. Missing files cause a clear error; there is no hub
download or cache fallback. `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` can also be
set as an additional offline guard. The loader reads only the requested split
and retains the original question order, answer corrections, and IDs. Redux
subjects are concatenated in the established 57-subject order.

The prepared data used for validation came from these immutable upstream
dataset snapshots:

| Dataset repository | Revision | Required evaluation/calibration sizes |
| --- | --- | --- |
| `cais/mmlu` | `c30699e8356da336a370243923dbaf21066bb9fe` | validation 1,531; test 14,042 |
| `edinburgh-dawg/mmlu-redux-2.0` | `372ea425445d51e1ba1188c56e5e893f8138621f` | 57 subjects × 100 test records |
| `tau/commonsense_qa` | `94630fe30dad47192a8546eb75f094926d47e155` | train 9,741; validation 1,221 |

Preparation preserves upstream row order and schemas. Save each individual
`datasets.Dataset` with `save_to_disk()` at the split path shown above; a raw
download directory or a Hugging Face cache is not a prepared split. MMLU uses
the upstream `all` configuration, including its `subject` column. Redux uses
the separate subject packages with `question`, `choices`, `answer`,
`error_type`, and `correct_answer` intact; corrections are applied by the
existing loader. CommonSenseQA preserves `question`, `choices` (with `label`
and `text`), and `answerKey`. For a prepared `DatasetDict`, saving it at the
configuration directory produces the same per-split layout.

MMLU `dev` (285 records) is additionally needed for few-shot prompts. CSQA
few-shot prompts use its training split. CSQA public `test` has no labels and
is not the held-out labeled evaluation split for this protocol.

Alternatively, pass `--data_file` with already formatted JSONL records:

```json
{"id": 0, "task_query": "What is 2 + 2?", "choices": ["3", "4", "5", "6"], "gold": 1, "subject": "elementary_mathematics"}
```

CommonSenseQA records have five choices and do not require `subject`.
Few-shot runs additionally need the original MMLU dev or CommonSenseQA train
examples; `--data_file` replaces the prediction input only.

## Result directories

Keep every new run under a unique `results/<run_id>/` directory. Run the commands
from the repository root and pass the paths explicitly; legacy CLI defaults remain available
for compatibility. The following shell variables only select paths and do not
create directories or start an experiment:

```bash
RUN_DIR="./results/example-run-001"
PREDICTIONS_DIR="$RUN_DIR/predictions"
ANALYSIS_DIR="$RUN_DIR/analysis"
```

The directories are created when their corresponding stages are run:

```text
results/<run_id>/
├── predictions/<dataset>/<split>/
│   ├── LLMs/confidence/logprob/<model>/...
│   └── post_processing/<method>/...
├── analysis/
│   ├── main/
│   ├── all_baselines/predictions/...
│   ├── standard_errors/
│   └── <experiment>/
└── logs/
```

Inference and post-processing share `PREDICTIONS_DIR`, including phase-1
intermediate records. Main and supplementary experiments read that directory
and write reports to `ANALYSIS_DIR/<experiment>`. Exported baseline predictions
remain under `analysis/all_baselines/predictions/`, which is the input to
standard-error calculations. The standalone score script keeps its existing
`confidence` to `score` path substitution within the predictions tree.

When starting a run, create `logs/` and save each stage's stdout/stderr there,
along with the exact command, code version or working-tree diff, seeds, and
local model/data versions. Use a fresh run ID for a rerun; do not reuse an old
batch directory or write generated tables into `src/`. The entire `results/`
tree is ignored by Git. Diagnostic runs should use their own run IDs as well.

The complete protocol uses all three model pairs and five dataset/split inputs:
MMLU validation and test,
MMLU-Redux test, and CommonSenseQA train and validation. For each pair and input,
run `base` with `baseline`, then `thinking_phase1` and `thinking_phase2` with
`baseline_thinking`: 45 stage invocations in total, or 15 per selected pair.
MMLU validation
is shared by the MMLU and MMLU-Redux calibrations, so generate it only once per
model. Phase 2 reads the saved phase-1 records. These templates do not require
few-shot dev examples. Prepare the local Arrow packages or formatted JSONL inputs
before starting; checkpoint availability alone does not supply the datasets.

## Inference

In the validated CUDA/GPU environment, FlashInfer's sampler JIT was
incompatible with the installed toolkit/GPU combination. The validated runs
used vLLM's native sampler; set this before starting inference:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

Keep this setting when reproducing those runs. It is a runtime configuration,
not a change to prompts, generation limits, or confidence extraction.

The shared entry point supports `base`, `instruct`, `thinking_phase1`, and
`thinking_phase2`. The dataset-specific scripts remain thin compatibility
entry points, and all options now come from the actual command line.

```bash
python -m source.LLM_predictions.confidence.logprob.pipeline \
  --dataset_name MMLU --split validation --mode base \
  --model_name mistralai/Ministral-3-8B-Base-2512 \
  --confidence_prompt_template baseline --output_dir "$PREDICTIONS_DIR"

python -m source.LLM_predictions.confidence.logprob.pipeline \
  --dataset_name MMLU --split validation --mode thinking_phase1 \
  --model_name mistralai/Ministral-3-8B-Instruct-2512 \
  --confidence_prompt_template baseline_thinking --output_dir "$PREDICTIONS_DIR"

python -m source.LLM_predictions.confidence.logprob.pipeline \
  --dataset_name MMLU --split validation --mode thinking_phase2 \
  --model_name mistralai/Ministral-3-8B-Instruct-2512 \
  --confidence_prompt_template baseline_thinking --output_dir "$PREDICTIONS_DIR"
```

Repeat for each dataset/split and model pair in the selected run. Use `--mode instruct` and
`--confidence_prompt_template baseline` for direct instruct scoring.
`--limit 2 --max_model_len 2048` bounds a smoke test; these limits are optional
and do not replace full experiment settings. `--thinking_max_tokens` defaults
to the original 1024 tokens. The default direct-scoring template remains
`5_shot`; specify `baseline` explicitly when no few-shot examples are intended.

Prompts under `source/LLM_predictions/prompts/` retain their original contents.
The existing Redux direct-scoring path uses MMLU templates. Thinking extraction
receives the dataset's legal options: ABCD or ABCDE. The E-option correction
retains the existing ABCD filtering, sentence deletion and think-tag rules.
New phase-1 records also save `raw_generation`, `filter_version`, `legal_options`
and length/empty diagnostics; phase 2 uses only the existing thinking and
conversation fields. Empty thinking still proceeds to phase 2. Log-probability
aggregation, top-20 scoring and missing-option handling are unchanged. An omitted seed
retains the previous engine defaults; an explicit `--seed` is now forwarded.
Both four-choice and five-choice scoring retain their original schemas.

Results retain the layout:

```text
<output_dir>/<dataset>/<split>/LLMs/confidence/logprob/<model_slug>/<template>.jsonl
<output_dir>/<dataset>/<split>/LLMs/confidence/logprob/<model_slug>/intermediate/<template>_phase1.jsonl
```

Existing inference files are protected; choose a new output directory or
explicitly supply `--overwrite`. Phase 2 reads phase 1 from the same output root.
Ministral uses vLLM's native Mistral tokenizer and text-only mode. This supports
the checkpoint's chat protocol without loading its unused image processor;
token IDs were checked against the original tokenizer for all three datasets
and chat modes. The loader also carries the checkpoint's embedding-sharing,
YaRN scaling, and long-context query-scaling metadata from `params.json` into
vLLM. This prevents library defaults from changing the saved model's behavior.
Checkpoint files and tokenizer normalization settings are unchanged.

## Calibration and experiments

The main and supplementary experiments share model/data configuration, record
loading, metrics, baseline implementations, and calibration adapters under
`src/experiments/`. Select only the desired pairs and datasets:

```bash
python -m src.experiments main --models qwen llama ministral \
  --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/main"

python -m source.post_processing.post_processing \
  --dataset_name MMLU --calibration_split validation \
  --application_dataset_name MMLU_Redux --application_split test \
  --model1_name mistralai/Ministral-3-8B-Instruct-2512 \
  --model2_name mistralai/Ministral-3-8B-Base-2512 --bin_width 0.01 \
  --confidence_dir "$PREDICTIONS_DIR" --output_dir "$PREDICTIONS_DIR"

python -m src.experiments all-baselines --models qwen llama ministral \
  --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/all_baselines"
```

The post-processing step prepares the `Ours` predictions read by the baseline
comparison. The main experiment uses the same mapping rule but does not create
that file. If the file is missing, the comparison warns and
omits that method. The post-processing command above illustrates Ministral on
MMLU-Redux; repeat it for every selected pair and dataset using the corresponding
split and model names before running the full baseline comparison.

Other commands are `label-free`, `dummy-reference`, `precision`,
`equal-mass-ece`, `failure-analysis`, `supervised-at-nstar`,
`sample-complexity`, and `reference-validation`. Each has `--help`.
Legacy experiment script paths remain wrappers for the same implementations.
The dummy-reference experiment keeps its original MMLU-Redux/CommonSenseQA
default; `--datasets MMLU` selects the additional supported dataset explicitly.

All active Ours confidence transformations first quantize the input confidence
using the existing round-half-up rule, then look up the exact
`(choice, quantized_confidence)` key. A hit uses that key's learned mapping;
a miss keeps the original choice and the quantized confidence. For example,
with `s=0.01`, `(A, 0.7248)` becomes `(A, 0.72)`. If that key is absent, the
output remains `(A, 0.72)` even when `(A, 0.71)`, `(A, 0.73)` or `(B, 0.72)`
has a learned mapping. There is no nearest-neighbor search, interpolation or
approximate key match.

The input grid defaults to `s=0.01`; the precision study varies only that grid
over `0.1, 0.05, 0.02, 0.01, 0.005, 0.002`. Both mapping hits and identity
fallbacks produce a numeric confidence rounded to four decimal places,
independently of `s`. Internal distributions, maps and statistics retain their
precision. Identity fallback retains the quantized confidence, not the
confidence before quantization.

Record selection remains specific to each entry point: main retains absent
choices and invalid reports; canonical post-processing filters invalid answers
and skips choices absent from calibration; the precision study retains its
original pair filtering and absent-choice handling. Retained absent-choice
reports also keep their quantized confidence. This deployment rule does not
restore records that an entry point filters or skips, or change its existing
solver-failure handling. The genuine, dummy and shuffled-reference variants
all use the same rule. Existing support-miss statistics retain their original
counting sets; when those include filtered or skipped records, their overall
rate is not the fallback rate among final predictions.

Confidence solves write diagnostic sidecars beside canonical predictions or
under the main/precision report directory's `diagnostics/` folder. These record
the existing solver's settings, status, residuals, objective/gap when available,
full projected distributions, clipping/normalization corrections and unrounded
maps. Linear residuals use scale 1000 and squared epigraph residuals use scale
1000000. Diagnostics introduce no new solve, acceptance threshold, retry or
change to the existing failure and `optimal_inaccurate` handling.

Comparison temperature scaling minimizes the mean per-record binary NLL using
row-wise softmax. Its bounds, clipping, optimizer and deployment sigmoid remain
unchanged. Standalone and budget TS retain their own existing numerical details,
including different endpoint clipping. GPcalib and BCT implementations, GP
sample budgets, and random-reference seeds are unchanged. BCT retains access to
its reference probability inputs; TS, GPcalib, PBR and Ours do not gain them.

PBR uses a shared sparse variational GP with
`q(c) = sigmoid(g(c) - g(1-c))`. It minimizes posterior expected binary-vector
Brier (`2 * (correct - q)**2`) plus `alpha / n_fit * KL(posterior || prior)`.
The KL denominator is the complete training fold, regardless of minibatch size.
Calibration-only five-fold CV (shuffle, seed 42) selects alpha from
`0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1` using predictive Brier, then refits the
complete calibration pool: 1,531 records for MMLU/Redux and 9,741 for CSQA.
GPcalib separately retains its 1,531/1,500-record budgets. PBR preserves the
original answer and predicts the posterior mean correctness probability.

The fixed configuration in `src/experiments/pbr.py` is a repository adaptation
of [Fujisawa and Futami, Eq. (8)](https://proceedings.mlr.press/v267/fujisawa25a.html),
informed by the paper's public
[GP precursor](https://github.com/JonathanWenger/pycalib/blob/2ec97e02374d10b57b6a33d9ee9fddcfa09c11a7/pycalib/calibration_methods.py).
It is not presented as an exact reproduction of the authors' training setup:
the prior mean is `log(p)`, the fixed RBF lengthscale/variance are `0.5/1`,
and 32 fixed uniform inducing locations use a whitened full-covariance Gaussian
initialized at the prior. CPU float64 Adam uses learning rate `0.01`, 1,000
steps, batches of 256 and 100 MC samples. Endpoint clipping is `1e-7` and
inducing Cholesky jitter is `1e-6`; there is no white kernel or independent
predictive noise. Prediction uses 64-point Gauss-Hermite integration of the
correlated GP difference. Private RNGs leave other methods' randomness intact.
No additional dependency is required beyond the existing PyTorch environment.

Both comparison commands save `posterior.npz` and `metadata.json` under
`<output_dir>/artifacts/PBR/<dataset>/<model>/`. Sidecars include the complete
configuration, input IDs/hash, CV scores/seeds, refit status/timing and model
hash; tables distinguish `pbr_calib_n` from `gp_calib_n`. Existing PBR artifacts
are never overwritten. The loader rejects legacy histogram models and checks
the implementation/configuration/parameter hashes. Model reuse additionally
requires matching calibration input hash and seed. `--fast` only selects one
comparison group; it does not shorten PBR training. Use the unit tests for
small numerical validation.

Individual post-processing commands remain available:

```bash
python -m source.post_processing.post_processing --help
python -m source.post_processing.max_confidence_aggregation --help
python -m source.post_processing.supervised_baselines --help
```

To score a transformation output with `source/summarize+figures/calculate_scores.py`,
set `--method_type post_processing`, use `--dataset_name`/`--split` for the
evaluation data, and set `--calibration_dataset_name`/`--calibration_split` to
the calibration source. For example, the Redux example above uses
`--dataset_name MMLU_Redux --split test --calibration_dataset_name MMLU
--calibration_split validation`. Omitted calibration options use the evaluation
dataset and split, matching standalone post-processing. The script reads the
corresponding `calibrated_on_...` directory and never falls back to an unrelated
older prediction directory. For `histogram_binning`, `isotonic_regression`,
`temperature_scaling`, and `max_confidence`, select `--post_processing_method`;
the scorer uses those methods' output layout without a transformation type or
calibration tag.

The supervised post-processing command retains its original convention of
reading training inputs from `--output_dir`; other input/output conventions
are described in its CLI. Set both its `--confidence_dir` and `--output_dir` to
`"$PREDICTIONS_DIR"`. Standalone post-processing defaults to `test` for
both splits; explicitly set the calibration and application splits as above.
Confidence input binning defaults to `0.01`; legacy `--precision` selects only
the input grid (`10**(-precision)`), and `--bin_width` overrides it. Output
confidence always uses four places. The historical mode/full-prediction paths
retain their default precision 4. Standard-error and alternative-ECE scripts remain
under `src/calculating_standard_errors/` and share their common calculations.
`all-baselines` and `dummy-reference` write or replace dummy, shuffled, and
calibrated intermediate JSONL files beneath `--results_dir`; `--output_dir`
only separates their reports and exported predictions. Use an independent
results tree when validating changes against existing artifacts.

For standard errors or alternative ECE, pass
`--pred_dir "$ANALYSIS_DIR/all_baselines/predictions"` and
`--output_dir "$ANALYSIS_DIR/standard_errors"`. The separate SmoothECE audit
currently expects all registered model pairs and has no model filter; it is
not yet a supported analysis step for a batch containing only Qwen and Ministral.

Export uncertainty without replacing raw `se_*` columns:

```bash
python -m src.calculating_standard_errors.scale_se_to_ci \
  --input "$ANALYSIS_DIR/standard_errors/standard_errors.csv" \
  --input-type raw-se \
  --output "$ANALYSIS_DIR/standard_errors/standard_errors_ci95.csv"
```

The added `ci95_halfwidth_*` fields equal `1.96 * SE` with the existing
six-decimal rounding. For historical files whose `se_*` columns already contain
half-widths, explicitly select `--input-type legacy-ci95-halfwidth`; the exporter
retains those half-widths and marks raw SE unavailable. It never recovers raw SE
by dividing rounded values or infers the input type from a filename. Estimators,
fixed-map assumptions, SmoothECE common-ID sets, paired bootstrap and bandwidth
rules are unchanged.

The sample-complexity `simple` rule selects the first grid point whose empirical
2.5% ECE-gap quantile is positive. `strict` retains the existing bootstrap and
separation threshold. Reports distinguish calibration/evaluation pools, outer
subset repetitions, bootstrap repetitions, number of seeds and across-seed SD;
not-reached remains not-reached. Reproduce a batch from its saved configuration,
including the global model/pool indices used for seed assignment. The revision
does not change any statistical default, candidate grid or decision rule.

## Preparing inputs after the calibration revision

The correction from nearest-neighbor deployment to quantized identity fallback
changes only application of an Ours map. On unchanged inputs, regenerate Ours
main/canonical and reference-control predictions, all 54 input-grid ablation
settings, and their dependent metrics and comparisons. A saved projection map
can be reused only after verifying the input hashes, input grid `s`, solver
configuration and representation precision. A compatible map needs no new
projection solve, and this deployment correction alone needs no LLM inference.
Final predictions produced by the previous nearest-neighbor protocol cannot be
reused as corrected outputs. The input-preparation command below copies no
maps; eligible map reuse must be checked and recorded separately.

The cumulative revision also includes the E-option, TS and PBR corrections.
Their rerun requirements below remain in effect; they are not caused by the
deployment-only correction.

Use a new run for both `--results_dir` and `--output_dir`. This read-only command
validates counts, unique IDs and hashes, and prints a reuse plan:

```bash
python -B -m src.experiments.revision_inputs \
  --source-predictions ./results/previous-run/predictions \
  --run-dir ./results/calibration-revision-new
```

Add `--copy` only when preparing that new run. It requires a nonexistent
destination and copies a fixed allowlist: all 15 Base prediction files, nine
MMLU/Redux Instruct prediction files and their nine phase-1 files. It copies no
CSQA Instruct outputs, derived predictions, maps, models or statistical caches.
Its manifest records input hashes, code/environment information and the six
model/split combinations awaiting phase 1 and phase 2. This tool starts no
inference, calibration or statistics.

CSQA needs new phase 1 and phase 2 for the Qwen, LLaMA and Ministral Instruct
models on train (9,741 records) and validation (1,221 records): 12 stages total.
The old phase-1 records lack the raw text needed to apply the E correction
retroactively. Preserve the original checkpoint/data revisions, prompts,
generation configuration, execution order and seed forwarding. In particular,
an omitted seed must remain omitted. The local Ministral Instruct checkpoint
uses the BF16 directory listed above; its public model identifier is unchanged.

After these inputs are ready, the rerun dependencies are:

| Output | MMLU / MMLU-Redux | CommonSenseQA |
| --- | --- | --- |
| Base raw predictions and native metrics | Reuse | Reuse |
| Instruct raw predictions and native metrics | Reuse | Regenerate predictions, recompute metrics |
| Comparison TS and PBR | Refit and predict | Refit and predict with new inputs |
| GP, BCT, ConfAvg, Ensemble, HB, IR, DACA | Reuse if inputs/settings match | Refit or apply with new Instruct inputs; GP budget stays 1,500 |
| Ours main/canonical and reference controls | Regenerate predictions with exact-key mapping/quantized identity and four-place output; reuse verified compatible maps | Fit on new inputs, then use the corrected deployment protocol and diagnostics |
| Input-grid ablation | Regenerate predictions and metrics for all six widths; compatible maps need no new solve | Fit on new inputs for all six widths; 18 of the 54 settings across all nine groups |
| Metrics, SE/CI, alternative/equal-mass ECE, failure analysis | Refresh affected predictions | Refresh new-input dependencies |
| SmoothECE | Recompute all nine groups with unchanged protocol | Recompute every method, including Base, on the new common IDs |
| Few-label N* | Reuse matching original-input results | Recompute six model/pool pairs: single-seed simple/strict and existing multi-seed simple |
| Supervised-at-N* auxiliary analysis | Reuse supervised predictions where eligible; refresh differences to Ours | Refit with new inputs and corresponding budgets |

Historical fixed budgets remain labeled historical; current simple/strict
budgets come from the calibration split's N* results. Do not replace a
not-reached budget with a finite value or add multi-seed strict experiments.
The comparison paths write some intermediate predictions under `--results_dir`,
so changing only the report directory does not isolate a rerun.

Reuse is justified by matching input hashes, method protocol, settings and
seeds, rather than by the overall code commit alone. Record actual commands,
model/data revisions, method configuration, seeds, elapsed time and reuse
sources in each new batch. For Ours, record `s`, fixed four-place output, the
exact-key/quantized-identity deployment protocol, and the provenance of any
reused map. Matching artifact names alone do not establish compatibility;
earlier predictions with a different deployment protocol or output precision
are not reusable Ours outputs. Keep the original batch intact, and check complete
method coverage and batch consistency after the new run. Do not treat results
derived from old CSQA predictions as final revised results.

## Validation and layout

```bash
python -B -m unittest discover -s tests -v
```

Tests run on the CPU without model downloads or full experimental fits. They
cover preserved prompt/sampling behavior, the authorized TS/PBR/Ours changes,
diagnostic noninterference, uncertainty exports, input isolation, dataset
formatting and local checkpoint validation.

- `data/`: dataset loading and formatting.
- `source/LLM_predictions/`: prompt assets and shared inference pipeline.
- `source/post_processing/`: prediction transformations and supervised methods.
- `src/calibration.py`, `src/metrics.py`: main calibration and evaluation.
- `src/experiments/`: experiment implementations and shared components.
- `src/calculating_standard_errors/`: uncertainty and alternative metric tools.
- `tests/`: deterministic regression tests.

Generated predictions, tables, plots, downloaded datasets, and model weights
remain local ignored artifacts. They are not part of the source release.
Keep generated files under `results/` (including prediction JSONL, diagnostic
JSON, posterior NPZ, tables, and plots) so ignore rules cover them together.
Prompt templates, tests, dependency files, and method documentation remain
tracked. `.gitignore` also excludes local environments, credentials, caches,
logs, and checkpoint files; it does not sanitize a file already tracked by Git
or erase earlier commits. Review both the exact staged tree and the history of
the branch selected for publication before pushing it publicly. Do not publish
local runtime manifests or logs without reviewing their paths and metadata.
