"""Per-example NLL and supervised fits for the N-star label-budget protocol."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from src.metrics import is_correct
from source.post_processing.supervised_baselines import HistogramBinning, TemperatureScaling

def conf_to_binary_logits(confs, eps=1e-12):
    """
    Helper function: Convert probability p into binary logits [log(p), log(1-p)]
    """
    logits_list = []
    for p in confs:
        # Clip to prevent numerical errors
        p = float(np.clip(p, eps, 1 - eps))
        # Construct [logit_correct, logit_incorrect]
        logit_vec = np.array([np.log(p), np.log(1 - p)])
        logits_list.append(logit_vec)
    return logits_list

def fit_apply_ts(calib_records, test_records):
    """
    Temperature Scaling (binary logit mode via TemperatureScaling class).
    """
    train_confs  = np.array([float(r["model_prediction"]["confidence"]) for r in calib_records])
    train_corr   = np.array([int(is_correct(r)) for r in calib_records])

    # 1. Prepare training data
    train_binary_logits = conf_to_binary_logits(train_confs)
    # Labels: index 0 represents "Correct", index 1 represents "Incorrect"
    train_binary_labels = [0 if c else 1 for c in train_corr]

    # 2. Fit the model
    model = TemperatureScaling()
    model.fit(train_binary_logits, train_binary_labels)

    # 3. Predict on test records
    test_confs = np.array([float(r["model_prediction"]["confidence"]) for r in test_records])
    test_binary_logits = conf_to_binary_logits(test_confs)

    calibrated = []
    for i, rec in enumerate(test_records):
        pred_probs = model.predict(test_binary_logits[i])
        c_new = float(pred_probs[0]) # Prob of the "Correct" surrogate class

        new_rec = dict(rec)
        new_rec["model_prediction"] = dict(rec["model_prediction"])
        new_rec["model_prediction"]["confidence"] = c_new
        calibrated.append(new_rec)

    return calibrated, float(model.temperature)

def fit_apply_hb(calib_records, test_records, n_bins=10):
    """
    Histogram Binning (via HistogramBinning class).
    Assigns each test sample the mean accuracy of its calibration bin.
    """
    train_confs  = np.array([float(r["model_prediction"]["confidence"]) for r in calib_records])
    train_corr   = np.array([int(is_correct(r)) for r in calib_records])

    # Fit
    model = HistogramBinning(n_bins=n_bins)
    model.fit(train_confs, train_corr)

    # Predict
    test_confs = np.array([float(r["model_prediction"]["confidence"]) for r in test_records])
    new_confs = model.predict(test_confs)

    calibrated = []
    for i, rec in enumerate(test_records):
        new_rec = dict(rec)
        new_rec["model_prediction"] = dict(rec["model_prediction"])
        new_rec["model_prediction"]["confidence"] = float(new_confs[i])
        calibrated.append(new_rec)

    return calibrated

def fit_apply_ir(calib_records, test_records):
    """
    Isotonic Regression (using sklearn class matching the module data flow).
    """
    train_confs  = np.array([float(r["model_prediction"]["confidence"]) for r in calib_records])
    train_corr   = np.array([int(is_correct(r)) for r in calib_records])

    # Fit
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(train_confs, train_corr)

    # Predict
    test_confs = np.array([float(r["model_prediction"]["confidence"]) for r in test_records])
    new_confs = model.predict(test_confs)

    calibrated = []
    for i, rec in enumerate(test_records):
        new_rec = dict(rec)
        new_rec["model_prediction"] = dict(rec["model_prediction"])
        # round up similarly or directly cast float
        new_rec["model_prediction"]["confidence"] = float(new_confs[i])
        calibrated.append(new_rec)

    return calibrated
