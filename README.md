# Calibration without Ground Truth

This repository contains the official implementation of our paper [Calibration without Ground Truth](https://arxiv.org/abs/2601.19862), accepted as a **NeurIPS 2026 Spotlight (1%)**.

## Conda environment

Run the commands below from the repository root in the same Bash session.

```bash
conda create -n calibration python=3.11.15 -y
conda activate calibration
python -m pip install -r requirements.txt
python -m pip check
```

| Package | Version |
| --- | --- |
| Python | 3.11.15 |
| numpy | 2.3.5 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| cvxpy | 1.9.3 |
| clarabel | 0.11.1 |
| osqp | 1.1.3 |
| matplotlib | 3.11.2 |
| tqdm | 4.70.0 |
| datasets | 5.0.1 |
| torch | 2.11.0 |
| transformers | 5.13.0 |
| vllm | 0.26.0 |
| mistral-common | 1.11.7 |

`requirements.txt` includes the dependencies for the full experiment pipeline.

## Hugging Face inputs

### Model downloads

Download the six checkpoints below from Hugging Face and save each complete
checkpoint in the indicated directory under `MODEL_ROOT`. With the
`MODEL_ROOT="$PWD/models"` setting used below, these directories are under
`models/` in the repository root. Include the weights, configuration, and
tokenizer files, and complete the downloads before enabling the offline settings
in the next section. Obtain access to gated repositories where required.

| Hugging Face download repository | CLI identifier | Directory under `MODEL_ROOT` |
| --- | --- | --- |
| [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | `Qwen/Qwen3-8B-Base` | `Qwen3-8B-Base/` |
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `Qwen/Qwen3-8B` | `Qwen3-8B/` |
| [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | `meta-llama/Llama-3.1-8B` | `Llama-3.1-8B/` |
| [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `meta-llama/Llama-3.1-8B-Instruct` | `Llama-3.1-8B-Instruct/` |
| [mistralai/Ministral-3-8B-Base-2512](https://huggingface.co/mistralai/Ministral-3-8B-Base-2512) | `mistralai/Ministral-3-8B-Base-2512` | `Ministral-3-8B-Base-2512/` |
| [mistralai/Ministral-3-8B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16) | `mistralai/Ministral-3-8B-Instruct-2512` | `Ministral-3-8B-Instruct-2512-BF16/` |

For Ministral Instruct, download the **BF16** checkpoint while retaining the
CLI identifier without `-BF16` shown above. The code maps that identifier to the
BF16 directory; use the CLI identifier in the experiment commands.

### Dataset sources and splits

| Dataset | CLI identifier | Dataset repository | Calibration split | Evaluation split |
| --- | --- | --- | --- | --- |
| MMLU | `MMLU` | `cais/mmlu` | MMLU `validation` | `test` |
| MMLU-Redux | `MMLU_Redux` | `edinburgh-dawg/mmlu-redux-2.0` | MMLU `validation` | `test` |
| CommonSenseQA | `CommonSenseQA` | `tau/commonsense_qa` | CommonSenseQA `train` | `validation` |

## Directories and paths

| Directory | Contents |
| --- | --- |
| `data/` | Dataset loading entry points |
| `source/LLM_predictions/` | Inference entry points and `prompts/` |
| `source/post_processing/` | Prediction export entry points |
| `src/experiments/` | Main and supplementary experiment entry points |
| `src/calculating_standard_errors/` | Metric and uncertainty export commands |
| `tests/` | Tests |
| `models/` | Local model checkpoints; ignored by Git |
| `datasets/` | Local prepared datasets; ignored by Git |
| `results/<run_id>/` | Predictions, reports, tables, figures, and logs; ignored by Git |

Set the local input paths and choose a new result directory:

```bash
set -euo pipefail
export MODEL_ROOT="$PWD/models"
export DATASET_ROOT="$PWD/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0

RUN_DIR="$PWD/results/experiment-001"
PREDICTIONS_DIR="$RUN_DIR/predictions"
ANALYSIS_DIR="$RUN_DIR/analysis"
```

| Variable | Meaning |
| --- | --- |
| `MODEL_ROOT` | Directory containing the six checkpoint directories listed above |
| `DATASET_ROOT` | Directory containing the prepared dataset paths below |
| `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` | Disable online input lookup |
| `VLLM_USE_FLASHINFER_SAMPLER` | Inference runtime setting; use `0` for these commands |
| `RUN_DIR` | Root of one complete experiment run |
| `PREDICTIONS_DIR` | Input/output root for inference and post-processing predictions |
| `ANALYSIS_DIR` | Output root for experiment reports and metric summaries |

Both input roots may point outside the repository. Without overrides, they are
`CODE_ROOT.parent.parent / "models"` and `CODE_ROOT.parent.parent / "datasets"`,
where `CODE_ROOT` is the directory containing this README. Model path precedence
is `--model_path`, `--model_root`, `MODEL_ROOT`, then the default root.

Prepared dataset split directories must be saved with
`datasets.Dataset.save_to_disk()` and contain `state.json`, `dataset_info.json`,
and their Arrow files:

```text
<DATASET_ROOT>/
├── MMLU/prepared/all/{validation,test}/
├── MMLU-Redux-2.0/prepared/<subject>/test/
└── CommonSenseQA/prepared/default/{train,validation}/
```

The model checkpoints and prepared datasets must already exist before step 1.
The result layout is:

```text
results/<run_id>/
├── predictions/<dataset>/<split>/
│   ├── LLMs/confidence/logprob/<model_slug>/<template>.jsonl
│   ├── LLMs/confidence/logprob/<model_slug>/intermediate/<template>_phase1.jsonl
│   ├── post_processing/...
│   └── LLMs/score/... and post_processing/.../score/...
├── analysis/
│   ├── main/
│   ├── all_baselines/{predictions,artifacts}/
│   ├── label_free/
│   ├── dummy_reference/
│   ├── precision/
│   ├── equal_mass_ece/
│   ├── failure_analysis/
│   ├── sample_complexity/
│   ├── supervised_at_nstar/
│   ├── standard_errors/
│   └── smooth_ece/
└── logs/
```

`model_slug` is the model identifier with `/` replaced by `_`. Use a fresh
`RUN_DIR` for every full run. Subsequent steps reuse the predictions from that run.

## Full experiment pipeline

### 1. Inference and raw-prediction scores

```bash
MODEL_PAIRS=(
  'Qwen/Qwen3-8B Qwen/Qwen3-8B-Base'
  'meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-8B'
  'mistralai/Ministral-3-8B-Instruct-2512 mistralai/Ministral-3-8B-Base-2512'
)
INPUTS=(
  'MMLU validation'
  'MMLU test'
  'MMLU_Redux test'
  'CommonSenseQA train'
  'CommonSenseQA validation'
)

for pair in "${MODEL_PAIRS[@]}"; do
  read -r INSTRUCT_MODEL BASE_MODEL <<< "$pair"
  for input in "${INPUTS[@]}"; do
    read -r DATASET SPLIT <<< "$input"
    python -m source.LLM_predictions.confidence.logprob.pipeline \
      --dataset_name "$DATASET" --split "$SPLIT" --mode base \
      --model_name "$BASE_MODEL" --confidence_prompt_template baseline \
      --temperature 0 --max_tokens 1 --output_dir "$PREDICTIONS_DIR"

    for PHASE in thinking_phase1 thinking_phase2; do
      python -m source.LLM_predictions.confidence.logprob.pipeline \
        --dataset_name "$DATASET" --split "$SPLIT" --mode "$PHASE" \
        --model_name "$INSTRUCT_MODEL" \
        --confidence_prompt_template baseline_thinking \
        --temperature 0 --thinking_max_tokens 1024 --max_tokens 1 \
        --output_dir "$PREDICTIONS_DIR"
    done

    for MODEL in "$BASE_MODEL" "$INSTRUCT_MODEL"; do
      python -m source.summarize+figures.calculate_scores \
        --dataset_name "$DATASET" --split "$SPLIT" \
        --confidence_dir "$PREDICTIONS_DIR" \
        --method_type LLMs --llm_method_name logprob --llm_model_name "$MODEL"
    done
  done
done
```

| Parameter | Meaning |
| --- | --- |
| `--dataset_name`, `--split` | Dataset and input split |
| `--model_name` | Model identifier from the supported-model table |
| `--mode` | `base`, `thinking_phase1`, or `thinking_phase2` in this pipeline |
| `--confidence_prompt_template` | Prompt directory name: `baseline` or `baseline_thinking` |
| `--temperature` | Generation temperature |
| `--thinking_max_tokens` | Phase-1 generation token limit |
| `--max_tokens` | Answer-scoring generation token limit |
| `--output_dir` | Prediction output root; phase 2 also reads phase 1 from this root |
| `--confidence_dir` | Prediction input root for scoring |
| `--method_type`, `--llm_method_name`, `--llm_model_name` | Prediction source category, scoring method identifier, and model identifier |

The loop runs 45 inference stages. Optional inference parameters include
`--model_root` or `--model_path` for checkpoint paths, `--data_file` for a prepared
JSONL input, and `--seed` for an explicit random seed. The commands above leave
`--seed` unset. Full runs omit `--limit`, `--debug`, and `--overwrite`.

### 2. Main experiment and prediction exports

```bash
python -m src.experiments main \
  --models qwen llama ministral --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/main"

CALIBRATION_JOBS=(
  'MMLU validation MMLU test'
  'MMLU validation MMLU_Redux test'
  'CommonSenseQA train CommonSenseQA validation'
)

for pair in "${MODEL_PAIRS[@]}"; do
  read -r INSTRUCT_MODEL BASE_MODEL <<< "$pair"
  for job in "${CALIBRATION_JOBS[@]}"; do
    read -r CALIB_DATASET CALIB_SPLIT DATASET SPLIT <<< "$job"

    python -m source.post_processing.post_processing \
      --dataset_name "$CALIB_DATASET" --calibration_split "$CALIB_SPLIT" \
      --application_dataset_name "$DATASET" --application_split "$SPLIT" \
      --model1_name "$INSTRUCT_MODEL" --model2_name "$BASE_MODEL" \
      --post_processing_type confidence --bin_width 0.01 \
      --confidence_dir "$PREDICTIONS_DIR" --output_dir "$PREDICTIONS_DIR"

    for METHOD in temperature_scaling histogram_binning isotonic_regression; do
      python -m source.post_processing.supervised_baselines \
        --dataset_name "$DATASET" --split "$SPLIT" \
        --aux_dataset "$CALIB_DATASET" --aux_split "$CALIB_SPLIT" \
        --model1_name "$INSTRUCT_MODEL" --model2_name "$BASE_MODEL" \
        --baseline_method "$METHOD" \
        --confidence_dir "$PREDICTIONS_DIR" --output_dir "$PREDICTIONS_DIR"
    done

    python -m source.post_processing.max_confidence_aggregation \
      --dataset_name "$DATASET" --split "$SPLIT" \
      --model1_name "$INSTRUCT_MODEL" --model2_name "$BASE_MODEL" \
      --confidence_dir "$PREDICTIONS_DIR" --output_dir "$PREDICTIONS_DIR"

    for METHOD in transformation temperature_scaling histogram_binning isotonic_regression max_confidence; do
      python -m source.summarize+figures.calculate_scores \
        --dataset_name "$DATASET" --split "$SPLIT" \
        --confidence_dir "$PREDICTIONS_DIR" --method_type post_processing \
        --llm_model_name "$INSTRUCT_MODEL" --post_processing_method "$METHOD" \
        --post_processing_type confidence \
        --calibration_dataset_name "$CALIB_DATASET" --calibration_split "$CALIB_SPLIT"
    done
  done
done
```

| Parameter | Meaning |
| --- | --- |
| `--models` | Model-pair selectors: `qwen`, `llama`, `ministral` |
| `--datasets` | Dataset selectors: `MMLU`, `MMLU_Redux`, `CommonSenseQA` |
| `--results_dir` | Prediction root read by experiment commands |
| `--output_dir` | Report directory for `src.experiments`; prediction root for post-processing commands |
| `--model1_name`, `--model2_name` | Instruct and Base model identifiers, respectively |
| `--dataset_name`, `--calibration_split` | Calibration dataset/split for `post_processing` |
| `--application_dataset_name`, `--application_split` | Evaluation dataset/split for `post_processing` |
| `--post_processing_type` | Export type; use `confidence` |
| `--bin_width` | Input discretization width |
| `--dataset_name`, `--split` | Evaluation dataset/split for standalone baselines and scoring |
| `--aux_dataset`, `--aux_split` | Calibration dataset/split for standalone supervised baselines |
| `--baseline_method` | Standalone baseline selector: `temperature_scaling`, `histogram_binning`, `isotonic_regression` |
| `--post_processing_method` | Export category to score |
| `--calibration_dataset_name`, `--calibration_split` | Calibration path selectors for scoring transformation outputs |

Complete the prediction exports before step 3. Keep both `--confidence_dir`
and `--output_dir` set to `PREDICTIONS_DIR` for the standalone baseline commands.

### 3. Baseline comparisons

```bash
python -m src.experiments all-baselines \
  --models qwen llama ministral --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/all_baselines"

python -m src.experiments label-free \
  --models qwen llama ministral --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/label_free"
```

Both commands use the experiment selectors and directory arguments defined in
step 2. `all-baselines` also exports the predictions required by step 6 to
`$ANALYSIS_DIR/all_baselines/predictions/`.

| Baseline or comparison entry | Command |
| --- | --- |
| Base, Instruct, Ours | `all-baselines`, `label-free` |
| BCT (Base Confidence Transfer), Ensemble, ConfAvg (Confidence Averaging), DACA | `all-baselines`; also included in `label-free` |
| TS (Temperature Scaling) | `all-baselines`, `label-free`; standalone export in step 2 |
| HB (Histogram Binning), IR (Isotonic Regression) | `all-baselines`; standalone exports in step 2 |
| GPcalib (Gaussian Process Calibration), PBR (PAC-Bayes Recalibration) | `all-baselines`; also included in `label-free` |
| `Ours_RandomRef`, `Ours_CalibratedRef`, `Ours_ShuffledRandomRef` | `all-baselines`, `dummy-reference` |
| `Ours_RealBase` | `dummy-reference` |
| MaxConf (Maximum Confidence) | `source.post_processing.max_confidence_aggregation` in step 2 |

The comparison commands run their listed methods together; they do not expose
an individual-method selection argument.

### 4. Supplementary experiments

```bash
python -m src.experiments dummy-reference \
  --models qwen llama ministral --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/dummy_reference"

python -m src.experiments precision \
  --models qwen llama ministral --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/precision" \
  --bin_widths 0.1 0.05 0.02 0.01 0.005 0.002

python -m src.experiments equal-mass-ece \
  --models qwen llama ministral --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/equal_mass_ece"

python -m src.experiments failure-analysis \
  --models qwen llama ministral --datasets MMLU MMLU_Redux \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/failure_analysis"
```

The shared parameters have the meanings in step 2. `--bin_widths` lists the
input discretization widths for the precision experiment. `failure-analysis`
supports MMLU and MMLU-Redux only.

### 5. N* and fixed-budget baseline experiments

```bash
# Single-seed simple and strict results.
python -m src.experiments sample-complexity \
  --models qwen llama ministral --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/sample_complexity" \
  --n_reps 1000 --n_seeds 1 --run_bootstrap \
  --n_bootstrap 1000 --threshold_strict 0.8 --seed 42

# Multi-seed simple results.
python -m src.experiments sample-complexity \
  --models qwen llama ministral --datasets MMLU MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/sample_complexity" \
  --n_reps 1000 --n_seeds 1000 --seed 42

# TS, HB, and IR at the configured historical fixed budgets.
python -m src.experiments supervised-at-nstar \
  --models qwen llama ministral --datasets MMLU_Redux CommonSenseQA \
  --results_dir "$PREDICTIONS_DIR" --output_dir "$ANALYSIS_DIR/supervised_at_nstar" \
  --n_reps 200 --seed 42
```

| Parameter | Meaning |
| --- | --- |
| `--n_reps` | Subset repetitions per sample size and seed for `sample-complexity`; repetitions per fixed budget for `supervised-at-nstar` |
| `--n_seeds` | Number of seeds for `sample-complexity` |
| `--run_bootstrap` | Include strict results in single-seed mode |
| `--n_bootstrap` | Bootstrap repetitions per subset |
| `--threshold_strict` | Threshold for strict mode |
| `--seed` | Base random seed |

The sample-complexity summaries are `summary.json` and
`multi_seed_summary.json`. Multi-seed mode runs simple results only.
`supervised-at-nstar` uses its built-in fixed budgets; its CLI has no argument
for loading a newly generated N* result file.

### 6. Metrics and uncertainty exports

Run after `all-baselines` and `precision` have completed:

```bash
python -m src.calculating_standard_errors.calculate_standard_errors \
  --pred_dir "$ANALYSIS_DIR/all_baselines/predictions" \
  --output_dir "$ANALYSIS_DIR/standard_errors"

python -m src.calculating_standard_errors.scale_se_to_ci \
  --input "$ANALYSIS_DIR/standard_errors/standard_errors.csv" \
  --input-type raw-se \
  --output "$ANALYSIS_DIR/standard_errors/standard_errors_ci95.csv"

python -m src.calculating_standard_errors.calculate_alternative_ece \
  --pred_dir "$ANALYSIS_DIR/all_baselines/predictions" \
  --output_dir "$ANALYSIS_DIR/standard_errors"

python -m src.calculating_standard_errors.scale_precision_ablation_se_to_ci \
  --input "$ANALYSIS_DIR/precision/summary_table.csv" \
  --output "$ANALYSIS_DIR/precision/summary_table_ci95.csv"

python -m src.calculating_standard_errors.calculate_smooth_ece \
  --pred-dir "$ANALYSIS_DIR/all_baselines/predictions" \
  --summary-csv "$ANALYSIS_DIR/all_baselines/all_baselines.csv" \
  --output-dir "$ANALYSIS_DIR/smooth_ece" \
  --bootstrap 1000 --seed 20260725 --workers 4 --ci-level 0.95
```

| Parameter | Meaning |
| --- | --- |
| `--pred_dir`, `--pred-dir` | Root of the predictions exported by `all-baselines` |
| `--output_dir`, `--output-dir` | Output directory; SmoothECE requires a new or empty directory |
| `--input`, `--output` | Input and output CSV paths |
| `--input-type raw-se` | Input CSV contains raw standard errors |
| `--summary-csv` | Baseline summary CSV |
| `--bootstrap` | SmoothECE bootstrap repetition count |
| `--seed` | SmoothECE random seed |
| `--workers` | Number of parallel analysis tasks |
| `--ci-level` | Confidence level |

SmoothECE expects the complete three-model, three-dataset comparison exports.
All commands support `--help` for their full argument lists.

## License and third-party acknowledgments

Original code and documentation in this repository are released under the
[MIT License](LICENSE). Third-party components retain their respective licenses.
The adapted SmoothECE implementation in `src/third_party/relplot_smece.py` comes
from [Apple's ml-calibration](https://github.com/apple/ml-calibration) and is
covered by the included
[Apple license](src/third_party/APPLE_ML_CALIBRATION_LICENSE.txt).
External datasets and model checkpoints remain subject to their original
licenses and terms of use.
