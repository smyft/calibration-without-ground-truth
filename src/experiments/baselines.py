"""Baselines used by the full and supplementary comparisons.

Comparison temperature scaling minimizes mean per-example binary NLL.
The label-budget experiment retains its separate numerical implementation.
"""

import copy
import numpy as np
from scipy.special import softmax as scipy_softmax
from scipy.optimize import minimize_scalar
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.isotonic import IsotonicRegression
from src.metrics import is_correct

PBR_ALPHA_GRID = [0., 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

def _subsample_for_gp(records, max_n=None, seed=42):
    if max_n is None or len(records) <= max_n:
        return records
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(records), size=max_n, replace=False).tolist())
    return [records[i] for i in idx]

def get_full_dist(record, *, allow_uniform=True):
    choices = record["choices"]
    if allow_uniform and ("logprobs_raw" not in record or not record["logprobs_raw"]):
        K = len(choices)
        return {c: 1.0 / K for c in choices}
    raw = record["logprobs_raw"]
    logprobs = np.array([raw[c] for c in choices], dtype=np.float64)
    logprobs = np.clip(logprobs, -500.0, 0.0)
    probs = scipy_softmax(logprobs)
    return {c: float(p) for c, p in zip(choices, probs)}

def apply_bct(inst_records, base_records, *, allow_uniform=True):
    result = []
    for inst_rec, base_rec in zip(inst_records, base_records):
        inst_answer = inst_rec["model_prediction"]["final_answer"]
        base_dist = get_full_dist(base_rec, allow_uniform=allow_uniform)
        new_conf = base_dist.get(inst_answer, 1.0 / len(inst_rec["choices"]))
        rec = copy.deepcopy(inst_rec)
        rec["model_prediction"]["confidence"] = new_conf
        result.append(rec)
    return result

def apply_ensemble(inst_records, base_records, *, allow_uniform=True):
    result = []
    for inst_rec, base_rec in zip(inst_records, base_records):
        choices = inst_rec["choices"]
        inst_dist = get_full_dist(inst_rec, allow_uniform=allow_uniform)
        base_dist = get_full_dist(base_rec, allow_uniform=allow_uniform)
        avg = {c: (inst_dist.get(c, 0.0) + base_dist.get(c, 0.0)) / 2.0 for c in choices}
        pred = max(avg, key=avg.get)
        rec = copy.deepcopy(inst_rec)
        rec["model_prediction"]["final_answer"] = pred
        rec["model_prediction"]["confidence"] = float(avg[pred])
        result.append(rec)
    return result

def apply_conf_avg(inst_records, base_records):
    result = []
    for inst_rec, base_rec in zip(inst_records, base_records):
        new_conf = (float(inst_rec["model_prediction"]["confidence"]) +
                    float(base_rec["model_prediction"]["confidence"])) / 2.0
        rec = copy.deepcopy(inst_rec)
        rec["model_prediction"]["confidence"] = new_conf
        result.append(rec)
    return result

def fit_daca(calib_inst, calib_base):
    base_map = {r["id"]: r for r in calib_base}
    agree_logits, agree_probs = [], []
    for inst_rec in calib_inst:
        base_rec = base_map.get(inst_rec["id"])
        if base_rec is None:
            continue
        if inst_rec["model_prediction"]["final_answer"] != base_rec["model_prediction"]["final_answer"]:
            continue
        p_inst = float(np.clip(inst_rec["model_prediction"]["confidence"], 1e-7, 1 - 1e-7))
        p_base = float(np.clip(base_rec["model_prediction"]["confidence"], 1e-7, 1 - 1e-7))
        agree_logits.append(np.log(p_inst / (1 - p_inst)))
        agree_probs.append(p_base)
    n_agree = len(agree_logits)
    n_total = len(calib_inst)
    print(f"  DACA: {n_agree}/{n_total} agreement examples ({100 * n_agree / n_total:.1f}%)")
    if n_agree == 0:
        print("  DACA: no agreement examples — returning τ=1.0")
        return 1.0
    logits = np.array(agree_logits)
    probs  = np.array(agree_probs)

    def _loss(tau):
        q = 1.0 / (1.0 + np.exp(-logits / tau))
        return float(-np.mean(probs * np.log(q + 1e-12) + (1 - probs) * np.log(1 - q + 1e-12)))

    res = minimize_scalar(_loss, bounds=(0.01, 1e5), method="bounded")
    tau = float(res.x)
    print(f"  DACA: fitted τ = {tau:.4f}")
    return tau

def apply_temperature_scaling(T, records):
    result = []
    for rec in records:
        c = rec["model_prediction"]["confidence"]
        logit = np.log(np.clip(c, 1e-7, 1 - 1e-7) / (1 - np.clip(c, 1e-7, 1 - 1e-7)))
        r = copy.deepcopy(rec)
        r["model_prediction"]["confidence"] = float(1.0 / (1.0 + np.exp(-logit / T)))
        result.append(r)
    return result

def fit_histogram_binning(calib_records, n_bins=10):
    confs  = np.array([r["model_prediction"]["confidence"] for r in calib_records])
    labels = np.array([int(is_correct(r)) for r in calib_records])
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_scores = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (confs > boundaries[i]) & (confs <= boundaries[i + 1])
        if mask.sum() > 0:
            bin_scores[i] = float(labels[mask].mean())
        else:
            bin_scores[i] = (boundaries[i] + boundaries[i + 1]) / 2.0
    return {"boundaries": boundaries, "bin_scores": bin_scores, "n_bins": n_bins}

def apply_histogram_binning(hb_model, records):
    boundaries = hb_model["boundaries"]
    bin_scores = hb_model["bin_scores"]
    n_bins     = hb_model["n_bins"]
    result = []
    for rec in records:
        c   = float(rec["model_prediction"]["confidence"])
        idx = int(np.digitize(c, boundaries, right=True)) - 1
        idx = max(0, min(n_bins - 1, idx))
        r   = copy.deepcopy(rec)
        r["model_prediction"]["confidence"] = float(bin_scores[idx])
        result.append(r)
    return result

def fit_isotonic_regression(calib_records):
    confs  = np.array([r["model_prediction"]["confidence"] for r in calib_records])
    labels = np.array([int(is_correct(r)) for r in calib_records])
    model  = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(confs, labels)
    return model

def apply_isotonic_regression(ir_model, records):
    confs = np.array([r["model_prediction"]["confidence"] for r in records])
    new_confs = ir_model.predict(confs)
    result = []
    for rec, c_new in zip(records, new_confs):
        r = copy.deepcopy(rec)
        r["model_prediction"]["confidence"] = float(c_new)
        result.append(r)
    return result

def fit_temperature_scaling(calib_records):
    confs  = np.array([r["model_prediction"]["confidence"] for r in calib_records], dtype=np.float64)
    labels = np.array([float(is_correct(r)) for r in calib_records], dtype=np.float64)
    logits = np.log(np.clip(confs, 1e-7, 1 - 1e-7) / (1 - np.clip(confs, 1e-7, 1 - 1e-7)))

    def nll(T):
        scaled = scipy_softmax(np.stack([logits / T, np.zeros_like(logits)], axis=1), axis=1)[:, 0]
        return -np.mean(labels * np.log(scaled + 1e-12) + (1 - labels) * np.log(1 - scaled + 1e-12))

    return float(minimize_scalar(nll, bounds=(0.01, 1e5), method="bounded").x)

def fit_gpcalib(calib_records):
    confs  = np.array([r["model_prediction"]["confidence"] for r in calib_records])
    labels = np.array([int(is_correct(r)) for r in calib_records])
    kernel = RBF(length_scale=0.1, length_scale_bounds=(0.05, 10.0))
    gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    gpc.fit(confs.reshape(-1, 1), labels)
    return gpc

def apply_gpcalib(gpc, records):
    confs = np.array([r["model_prediction"]["confidence"] for r in records])
    probs = gpc.predict_proba(confs.reshape(-1, 1))[:, 1]
    result = []
    for rec, p in zip(records, probs):
        r = copy.deepcopy(rec)
        r["model_prediction"]["confidence"] = float(p)
        result.append(r)
    return result

def fit_pbr(calib_records, alpha_grid=PBR_ALPHA_GRID, n_bins=10, n_cv_folds=5,
            *, config=None, expected_n=None, artifact_dir=None):
    """Compatibility wrapper for the shared-GP PBR implementation.

    The legacy n_bins argument is accepted but unused: PBR is not histogram
    shrinkage. Passing config fixes the complete training protocol explicitly.
    """
    from src.experiments.pbr import PBRConfig, fit_pbr as fit_gp_pbr
    if config is None:
        config = PBRConfig(alpha_grid=tuple(alpha_grid), n_cv_folds=n_cv_folds)
    return fit_gp_pbr(calib_records, config=config, expected_n=expected_n,
                      artifact_dir=artifact_dir)


def apply_pbr(pbr_model, records):
    from src.experiments.pbr import apply_pbr as apply_gp_pbr
    return apply_gp_pbr(pbr_model, records)
