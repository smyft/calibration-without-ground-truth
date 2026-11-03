"""
Source-pinned SmoothECE implementation from ``relplot==1.0.3``.

This is a minimal adaptation of the reflected-Gaussian SmoothECE path in
Apple's ``ml-calibration`` repository.  Plotting code and unrelated metrics
are intentionally omitted so this experiment does not require installing the
full plotting package.

Upstream repository:
    https://github.com/apple/ml-calibration
Upstream commit:
    18ff21a7e4e409fc4885690129f50211b32ea144
Upstream source blobs:
    src/relplot/metrics.py  662821b962ea67c21515b1a133a7692ae6ac793d
    src/relplot/kernels.py  a55bec6793ef18ea735e58466e6d5143f5e5b660
    src/relplot/config.py   65059c7f156df78ef44b8bd43dfd85fda216ef8d

For licensing, see ``APPLE_ML_CALIBRATION_LICENSE.txt`` in this directory.
Copyright (C) 2023 Apple Inc. All Rights Reserved.
"""

import numpy as np


RELPlot_VERSION = "1.0.3"
RELPlot_COMMIT = "18ff21a7e4e409fc4885690129f50211b32ea144"
SMECE_MESH_POINTS = 200


def _validate_inputs(confidence, correctness):
    """Return validated one-dimensional float64 arrays."""
    f = np.asarray(confidence, dtype=np.float64)
    y = np.asarray(correctness, dtype=np.float64)
    if f.ndim != 1 or y.ndim != 1:
        raise ValueError("SmoothECE expects one-dimensional arrays.")
    if len(f) == 0 or len(f) != len(y):
        raise ValueError("SmoothECE inputs must be non-empty and equally sized.")
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(y)):
        raise ValueError("SmoothECE inputs must be finite.")
    if np.any((f < 0.0) | (f > 1.0)):
        raise ValueError("Confidence values must lie in [0, 1].")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("Correctness values must be binary.")
    return f, y


def _interpolate(t, y):
    buckets_count = len(y)
    bucket_size = 1 / (buckets_count - 1)
    indices = (t * (buckets_count - 1)).astype(int).clip(0, buckets_count - 2)
    residual = (t - indices * bucket_size) / bucket_size
    return y[indices] * (1 - residual) + y[indices + 1] * residual


def _smooth_round_to_grid(f, y, eval_points=1000):
    values = np.zeros(eval_points)
    bins = (f * (eval_points - 1)).astype(int).clip(0, eval_points - 2)
    fraction = f * (eval_points - 1) - bins
    np.add.at(values, bins, (1 - fraction) * y)
    np.add.at(values, bins + 1, fraction * y)
    return values


class _BaseKernelMixin:
    def smooth(self, f, y, x_eval, eps=0.0001):
        smoothed = self.apply(f, y, x_eval)
        density = self.apply(f, np.ones_like(y), x_eval) + eps
        smoothed = smoothed / density
        return smoothed, density


class _GaussianKernel(_BaseKernelMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def kernel_ev(self, num_eval_points):
        t = np.linspace(0, 1, num_eval_points)
        variance = self.sigma**2
        result = np.exp(-(t - 0.5) * (t - 0.5) / (2 * variance))
        result /= np.sqrt(2 * np.pi) * self.sigma
        return result

    def convolve(self, values, eval_points):
        kernel = self.kernel_ev(eval_points)
        return np.convolve(values, kernel, "same")

    def apply(self, f, y, x_eval, eval_points=None):
        if eval_points is None:
            eval_points = max(2000, round(20 / self.sigma))
        eval_points = (eval_points // 2) + 1
        values = _smooth_round_to_grid(f, y, eval_points=eval_points)
        smoothed = self.convolve(values, eval_points)
        return _interpolate(x_eval, smoothed)


class _ReflectedGaussianKernel(_GaussianKernel):
    def convolve(self, values, eval_points):
        kernel = self.kernel_ev(eval_points)
        extended = np.concatenate(
            [np.flip(values)[:-1], values, np.flip(values)[1:]]
        )
        return np.convolve(extended, kernel, "valid")[
            eval_points // 2 : eval_points // 2 + eval_points
        ]


def _smooth_ece_at_bandwidth_arrays(f, y, bandwidth):
    bandwidth = float(bandwidth)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("SmoothECE bandwidth must be finite and positive.")

    eval_points = max(round(10 / bandwidth), SMECE_MESH_POINTS)
    grid = np.linspace(0, 1, eval_points)
    kernel = _ReflectedGaussianKernel(bandwidth)
    residual, density = kernel.smooth(f, f - y, grid)
    return float(np.sum(np.abs(residual) * density) / np.sum(density))


def smooth_ece_at_bandwidth(confidence, correctness, bandwidth):
    """Compute SmoothECE at one fixed reflected-Gaussian bandwidth."""
    f, y = _validate_inputs(confidence, correctness)
    return _smooth_ece_at_bandwidth_arrays(f, y, bandwidth)


def _search_param(predicate, start=1, refine=10):
    if predicate(start):
        return start
    start, end = 1, 0
    for _ in range(refine):
        midpoint = (start + end) / 2
        if predicate(midpoint):
            end = midpoint
        else:
            start = midpoint
    return start


def smECE(confidence, correctness, eps=0.001, return_width=False):
    """
    Compute SmoothECE with the automatic bandwidth from relplot 1.0.3.

    The public name and argument behavior intentionally match
    ``relplot.smECE``.
    """
    f, y = _validate_inputs(confidence, correctness)

    def check_smooth_ece(alpha):
        return alpha < eps or alpha < _smooth_ece_at_bandwidth_arrays(f, y, alpha)

    bandwidth = _search_param(check_smooth_ece, start=1, refine=10)
    value = _smooth_ece_at_bandwidth_arrays(f, y, bandwidth)
    if return_width:
        return value, bandwidth
    return value


def sm_ece(confidence, correctness, eps=0.001, return_width=False):
    """PEP-8 alias for :func:`smECE`."""
    return smECE(confidence, correctness, eps=eps, return_width=return_width)


__all__ = [
    "RELPlot_COMMIT",
    "RELPlot_VERSION",
    "SMECE_MESH_POINTS",
    "smECE",
    "sm_ece",
    "smooth_ece_at_bandwidth",
]
