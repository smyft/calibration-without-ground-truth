"""Binary PAC-Bayes recalibration with a shared sparse variational GP.

Implements Fujisawa & Futami (ICML 2025), Eq. (8), specialized to (1-c,c):
E[2(z-sigmoid(g(c)-g(1-c)))**2] + alpha / n_fit * KL(q(u)||p(u)).
The GP conditional is shared by prior and posterior, so their joint KL is the
inducing-variable KL (also an upper bound on the marginal process KL). The two evaluations of g are correlated, not separate GPs.

Configuration below is this repository's fixed binary adaptation, NOT a claim
of reproducing unpublished PBR author defaults. The cited public GP precursor
uses log(p) prior mean, RBF initialization (.5,1), MC=100 and maxiter=1000.
Here the prior is fixed, inducing locations are uniform, q is full covariance,
and Adam minibatches replace its GPflow/Scipy optimization. The white-kernel
term is omitted; jitter regularizes only the inducing Cholesky factor and is
not added as independent predictive noise. Deterministic Gaussian
quadrature integrates the predictive sigmoid, including posterior variance.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn

from src.metrics import is_correct

PBR_ALPHA_GRID = (0., .01, .1, .25, .5, .75, .9, 1.)
OBJECTIVE = "expected_binary_vector_brier_plus_alpha_over_n_fit_KL"
IMPLEMENTATION = "binary-shared-svgp-pbr-v1"
SOURCES = {
    "paper": "https://proceedings.mlr.press/v267/fujisawa25a.html",
    "paper_equation": "4.2, Eq. (8); no Eq. (10) classification-loss term",
    "gp_code": "https://github.com/JonathanWenger/pycalib/blob/2ec97e02374d10b57b6a33d9ee9fddcfa09c11a7/pycalib/calibration_methods.py",
    "provenance": "Original PBR paper refers to this public GP precursor. No separate PBR repository was linked on the PMLR page or found among the author's public repositories at implementation time. These are fixed repository adaptation settings, not claimed author defaults.",
}


@dataclass(frozen=True)
class PBRConfig:
    seed: int = 42
    cv_seed: int = 42
    n_cv_folds: int = 5
    alpha_grid: tuple[float, ...] = PBR_ALPHA_GRID
    n_inducing: int = 32
    kernel_lengthscale: float = .5
    kernel_variance: float = 1.
    white_kernel_variance: float = 0.
    jitter: float = 1e-6
    endpoint_clip: float = 1e-7
    learning_rate: float = .01
    adam_betas: tuple[float, float] = (.9, .999)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.
    adam_amsgrad: bool = False
    steps: int = 1000
    batch_size: int = 256
    mc_samples: int = 100
    predictive_quadrature_points: int = 64
    prediction_batch_size: int = 1024
    dtype: str = "float64"
    device: str = "cpu"
    prior_mean: str = "log_probability"
    prior_kernel: str = "fixed_RBF"
    inducing_placement: str = "fixed_uniform_clipped_probability_interval"
    variational_family: str = "whitened_full_covariance_gaussian"
    initialization: str = "q_equals_prior"
    optimizer: str = "Adam"

    def __post_init__(self):
        for name in ("n_inducing", "steps", "batch_size", "mc_samples",
                     "predictive_quadrature_points", "prediction_batch_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_inducing < 2 or self.n_cv_folds < 2:
            raise ValueError("At least two inducing points and two CV folds are required")
        if not self.alpha_grid or any(a < 0 or not np.isfinite(a) for a in self.alpha_grid):
            raise ValueError("alpha_grid must be finite and nonnegative")
        if not 0 < self.endpoint_clip < .5:
            raise ValueError("endpoint_clip must lie in (0,.5)")
        if min(self.kernel_lengthscale, self.kernel_variance, self.jitter, self.learning_rate) <= 0:
            raise ValueError("Kernel parameters, jitter, and learning_rate must be positive")
        if self.white_kernel_variance != 0:
            raise ValueError("This fixed PBR prior has no white kernel")
        if (len(self.adam_betas) != 2 or any(not 0 <= beta < 1 for beta in self.adam_betas)
                or self.adam_eps <= 0 or self.adam_weight_decay != 0):
            raise ValueError("Invalid Adam parameters; no weight-decay penalty is allowed")
        if (self.dtype, self.device, self.prior_mean, self.prior_kernel,
            self.inducing_placement, self.variational_family, self.initialization, self.optimizer) != (
                "float64", "cpu", "log_probability", "fixed_RBF",
                "fixed_uniform_clipped_probability_interval", "whitened_full_covariance_gaussian",
                "q_equals_prior", "Adam"):
            raise ValueError("Unsupported PBR parameterization")

    @property
    def config_id(self):
        return _hash({"implementation": IMPLEMENTATION, "config": asdict(self)})


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def gaussian_kl(mean, scale_tril):
    """KL(N(mean,L L^T) || N(0,I)); whitened inducing prior is standard normal."""
    return .5 * (mean.square().sum() + scale_tril.square().sum() - mean.numel()
                 - 2 * torch.log(torch.diagonal(scale_tril)).sum())


def expected_binary_brier(probability_samples, labels):
    """Expected vector Brier; averaging predictions before loss is incorrect."""
    return 2 * (probability_samples - labels).square().mean()


def pac_bayes_objective(expected_brier, kl, alpha, n_fit):
    if n_fit <= 0:
        raise ValueError("n_fit must be the full, nonempty training fold size")
    return expected_brier + (float(alpha) / n_fit) * kl


class SparseBinaryGP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        z = torch.linspace(config.endpoint_clip, 1 - config.endpoint_clip,
                           config.n_inducing, dtype=torch.float64)
        self.register_buffer("inducing", z)
        kzz = self.kernel(z, z) + config.jitter * torch.eye(len(z), dtype=torch.float64)
        self.register_buffer("prior_cholesky", torch.linalg.cholesky(kzz))
        self.q_mean = nn.Parameter(torch.zeros(len(z), dtype=torch.float64))
        # exp(diagonal)=1 initially; unconstrained off-diagonal, positive diagonal.
        self.q_raw_tril = nn.Parameter(torch.zeros((len(z), len(z)), dtype=torch.float64))

    def kernel(self, x, y):
        return self.config.kernel_variance * torch.exp(
            -.5 * ((x[:, None] - y[None, :]) / self.config.kernel_lengthscale).square())

    def scale_tril(self):
        return torch.tril(self.q_raw_tril, diagonal=-1) + torch.diag(
            torch.exp(torch.diagonal(self.q_raw_tril)))

    def kl(self):
        return gaussian_kl(self.q_mean, self.scale_tril())

    def _features(self, x):
        return torch.linalg.solve_triangular(
            self.prior_cholesky, self.kernel(self.inducing, x), upper=False).T

    def difference_features(self, confidences):
        """Marginals of the *shared* GP difference, retaining the cross covariance.

        Subtract the two inducing features BEFORE computing the variance. The
        residual conditional variance includes k(c,1-c). For c=.5 the same GP
        value cancels exactly, giving zero variance and q=.5.
        """
        c = confidences.clamp(self.config.endpoint_clip, 1 - self.config.endpoint_clip)
        d = self._features(c) - self._features(1 - c)
        prior_variance = 2 * self.config.kernel_variance * (
            1 - torch.exp(-.5 * ((2 * c - 1) / self.config.kernel_lengthscale).square()))
        conditional_variance = (prior_variance - d.square().sum(dim=1)).clamp_min(0)
        prior_mean = torch.log(c) - torch.log1p(-c)
        return prior_mean, d, conditional_variance

    def moments_from_features(self, features):
        prior_mean, d, conditional_variance = features
        mean = prior_mean + d @ self.q_mean
        variance = conditional_variance + (d @ self.scale_tril()).square().sum(dim=1)
        return mean, variance

    def difference_moments(self, confidences):
        return self.moments_from_features(self.difference_features(confidences))

    def probability_samples(self, features, n_samples, generator):
        """Sample each correlated GP difference marginal independently.

        Independent draws across records unbiasedly estimate the additive
        expected empirical risk; they are not a joint GP function draw across
        all records. Within each record the g(c), g(1-c) covariance is retained.
        """
        mean, variance = self.moments_from_features(features)
        noise = torch.randn((n_samples, len(mean)), dtype=torch.float64, generator=generator)
        # Exact zero for c=.5; clamped sqrt prevents 0 * infinite derivative.
        sd = variance.clamp_min(torch.finfo(torch.float64).tiny).sqrt()
        return torch.sigmoid(mean[None, :] + sd[None, :] * noise)

    def predictive_mean(self, confidences):
        # One-dimensional Gaussian quadrature integrates the correlated latent
        # difference, not sigmoid(E[g(c)] - E[g(1-c)]).
        values = np.asarray(confidences, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("PBR confidences must be finite probabilities")
        nodes, weights = np.polynomial.hermite.hermgauss(self.config.predictive_quadrature_points)
        nodes = torch.as_tensor(nodes, dtype=torch.float64)
        weights = torch.as_tensor(weights / np.sqrt(np.pi), dtype=torch.float64)
        pieces = []
        with torch.no_grad():
            for start in range(0, len(values), self.config.prediction_batch_size):
                c = torch.as_tensor(values[start:start + self.config.prediction_batch_size])
                mean, variance = self.difference_moments(c)
                probs = torch.sigmoid(mean[:, None] + (2 * variance[:, None]).sqrt() * nodes)
                pieces.append((probs @ weights).numpy())
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64)


def _fit(confidences, labels, alpha, config, seed):
    """One fold/full fit. Every stochastic operation has a private generator."""
    started = time.monotonic()
    model = SparseBinaryGP(config)
    c = torch.as_tensor(confidences, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    n_fit = len(c)
    if n_fit == 0:
        raise ValueError("PBR cannot fit an empty calibration pool")
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        features = tuple(v.detach() for v in model.difference_features(c))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=tuple(config.adam_betas), eps=config.adam_eps,
                                 weight_decay=config.adam_weight_decay, amsgrad=config.adam_amsgrad)
    first_loss = last_loss = None
    for step in range(config.steps):
        # Uniform without-replacement minibatch: unbiased mean empirical risk.
        index = rng.choice(n_fit, size=min(config.batch_size, n_fit), replace=False)
        batch = tuple(v[index] for v in features)
        samples = model.probability_samples(batch, config.mc_samples, generator)
        risk = expected_binary_brier(samples, y[index])
        kl = model.kl()
        loss = pac_bayes_objective(risk, kl, alpha, n_fit)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite PBR objective at step {step}")
        optimizer.zero_grad()
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"Nonfinite PBR gradient at step {step}")
        optimizer.step()
        last_loss = float(loss.detach())
        if first_loss is None:
            first_loss = last_loss
    with torch.no_grad():
        final_kl = float(model.kl())
    if not np.isfinite(final_kl):
        raise FloatingPointError("Nonfinite fitted PBR posterior")
    return model, {"status": "completed_fixed_steps", "steps": config.steps,
                   "n_fit": n_fit, "alpha": float(alpha), "seed": seed,
                   "kl_coefficient": float(alpha) / n_fit, "final_kl": final_kl,
                   "first_minibatch_objective": first_loss,
                   "last_minibatch_objective": last_loss,
                   "elapsed_seconds": time.monotonic() - started}


def fit_pbr(calib_records, *, config=None, expected_n=None, artifact_dir=None):
    config = config or PBRConfig()
    if artifact_dir is not None:
        for name in ("posterior.npz", "metadata.json"):
            if (Path(artifact_dir) / name).exists():
                raise FileExistsError(f"PBR artifacts already exist in {artifact_dir}; choose a fresh output directory")
    if expected_n is not None and len(calib_records) != expected_n:
        raise ValueError(f"PBR expected the full calibration pool ({expected_n}), got {len(calib_records)}")
    if len(calib_records) < config.n_cv_folds:
        raise ValueError("PBR calibration pool is smaller than n_cv_folds")
    confidences = np.array([float(r["model_prediction"]["confidence"]) for r in calib_records])
    labels = np.array([float(is_correct(r)) for r in calib_records])
    if not np.isfinite(confidences).all() or np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("PBR calibration confidences must be finite probabilities; no records are filtered")
    ids = [r["id"] for r in calib_records]
    if len({_hash(i) for i in ids}) != len(ids):
        raise ValueError("PBR calibration IDs must be unique; no records are deduplicated")
    inputs = [{"id": i, "confidence": float(c), "correct": int(y)}
              for i, c, y in zip(ids, confidences, labels)]
    folds = list(KFold(n_splits=config.n_cv_folds, shuffle=True,
                       random_state=config.cv_seed).split(confidences))
    scores = []
    for alpha in config.alpha_grid:
        fold_scores, training = [], []
        for fold_index, (train, validation) in enumerate(folds):
            # Common random numbers across alpha candidates; no global RNG use.
            seed = config.seed + 1 + fold_index
            model, state = _fit(confidences[train], labels[train], alpha, config, seed)
            prediction = model.predictive_mean(confidences[validation])
            fold_scores.append(float(np.mean(2 * (prediction - labels[validation]) ** 2)))
            training.append(state)
        scores.append({"alpha": float(alpha), "mean_brier": float(np.mean(fold_scores)),
                       "fold_brier": fold_scores, "training": training})
    selected = min(scores, key=lambda item: item["mean_brier"])
    alpha = selected["alpha"]
    final_seed = config.seed + 100000
    model, state = _fit(confidences, labels, alpha, config, final_seed)
    metadata = {"implementation": IMPLEMENTATION, "pbr_objective": OBJECTIVE,
                "pbr_calib_n": len(calib_records), "pbr_alpha": alpha,
                "pbr_config_id": config.config_id, "seed": config.seed,
                "config": asdict(config), "sources": SOURCES,
                "calibration_hash": _hash(inputs), "calibration_ids": ids,
                "calibration_ids_hash": _hash(ids), "cv_scores": scores,
                "cv_folds": [{"train_ids_hash": _hash([ids[i] for i in train]),
                               "validation_ids": [ids[i] for i in val],
                               "n_fit": len(train), "n_validation": len(val)} for train, val in folds],
                "cv_loss": "posterior_predictive_binary_vector_Brier",
                "cv_selection": "minimum_mean_fold_brier_first_grid_entry_breaks_ties",
                "training_status": state["status"], "final_training": state,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "predictive_integrator": "deterministic_Gauss_Hermite_including_posterior_variance",
                "mc_sampling": "private_torch_Generator; correlated_GP_difference_marginals; independent_across_records",
                "minibatch_sampling": "private_numpy_Generator; uniform_without_replacement_per_step",
                "environment": {"numpy": np.__version__, "torch": torch.__version__}}
    result = {"gp": model, "best_alpha": alpha, "metadata": metadata}
    if artifact_dir is not None:
        result["artifact_paths"] = save_pbr(result, artifact_dir)
    return result


def apply_pbr(pbr_model, records):
    if "gp" not in pbr_model or pbr_model.get("metadata", {}).get("implementation") != IMPLEMENTATION:
        raise ValueError("PBR requires a shared GP posterior; legacy histogram models are unsupported")
    probabilities = pbr_model["gp"].predictive_mean(
        [r["model_prediction"]["confidence"] for r in records])
    result = copy.deepcopy(records)
    for record, probability in zip(result, probabilities):
        record["model_prediction"]["confidence"] = float(probability)
    return result


def save_pbr(model, directory):
    """Numeric posterior plus inspectable metadata; refuse overwriting artifacts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    parameter_path, metadata_path = directory / "posterior.npz", directory / "metadata.json"
    if parameter_path.exists() or metadata_path.exists():
        raise FileExistsError(f"PBR artifacts already exist in {directory}; choose a fresh output directory")
    arrays = {k: value.detach().cpu().numpy() for k, value in model["gp"].state_dict().items()}
    with parameter_path.open("xb") as f:
        np.savez_compressed(f, **arrays)
    metadata = copy.deepcopy(model["metadata"])
    metadata["posterior_sha256"] = hashlib.sha256(parameter_path.read_bytes()).hexdigest()
    metadata["posterior_file"] = parameter_path.name
    with metadata_path.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
        f.write("\n")
    return {"model": str(parameter_path), "metadata": str(metadata_path)}


def load_pbr(directory):
    """Load only this implementation's validated numeric artifacts (no pickle)."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("implementation") != IMPLEMENTATION:
        raise ValueError("Incompatible PBR artifact implementation")
    if metadata.get("source_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("PBR implementation source hash mismatch; validate before reusing this artifact")
    parameter_path = directory / "posterior.npz"
    if hashlib.sha256(parameter_path.read_bytes()).hexdigest() != metadata["posterior_sha256"]:
        raise ValueError("PBR posterior hash mismatch")
    config = PBRConfig(**metadata["config"])
    if config.config_id != metadata["pbr_config_id"]:
        raise ValueError("PBR configuration hash mismatch")
    gp = SparseBinaryGP(config)
    with np.load(parameter_path, allow_pickle=False) as arrays:
        gp.load_state_dict({key: torch.as_tensor(arrays[key]) for key in arrays.files}, strict=True)
    return {"gp": gp, "best_alpha": metadata["pbr_alpha"], "metadata": metadata}


def result_metadata(model):
    """Small CSV/JSON-compatible fields; full CV/config/IDs live in sidecars."""
    meta = model["metadata"]
    fields = {key: meta[key] for key in ("pbr_calib_n", "pbr_alpha", "pbr_objective", "pbr_config_id", "seed")}
    fields["pbr_training_status"] = meta["training_status"]
    fields["pbr_calibration_hash"] = meta["calibration_hash"]
    fields["pbr_model_artifact"] = model.get("artifact_paths", {}).get("model")
    fields["pbr_metadata_artifact"] = model.get("artifact_paths", {}).get("metadata")
    return fields
