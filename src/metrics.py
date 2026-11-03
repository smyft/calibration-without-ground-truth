"""
Evaluation metrics for calibration experiments.

All functions accept a list of JSONL records with schema:
    {"id": int, "choices": [str, ...], "gold": int,
     "model_prediction": {"final_answer": str, "confidence": float}}
"""

import numpy as np


def is_correct(record):
    return record["choices"][record["gold"]] == record["model_prediction"]["final_answer"]


def ece_uwb(records, n_bins=10):
    """Uniform-Width Binning ECE with 10 equal-width bins on [0, 1]."""
    confs   = np.array([float(r["model_prediction"]["confidence"]) for r in records])
    correct = np.array([float(is_correct(r)) for r in records])
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confs <= hi) if i == 0 else ((confs > lo) & (confs <= hi))
        if mask.sum() == 0:
            continue
        ece += abs(confs[mask].mean() - correct[mask].mean()) * mask.sum() / len(confs)
    return float(ece)


def brier_score(records):
    confs   = np.array([float(r["model_prediction"]["confidence"]) for r in records])
    correct = np.array([float(is_correct(r)) for r in records])
    return float(np.mean((confs - correct) ** 2))


def accuracy(records):
    return float(np.mean([is_correct(r) for r in records]))


def confidence_loss(records):
    """Paper Eq. (1): E[-1(y=ŷ) + (ĉ - 1(y=ŷ))²] = brier_score - accuracy."""
    return brier_score(records) - accuracy(records)


def compute_metrics(records):
    bs  = brier_score(records)
    acc = accuracy(records)
    return {
        "ece":      ece_uwb(records),
        "brier":    bs,
        "accuracy": acc,
        "cl":       bs - acc,
        "n":        len(records),
    }
