"""Small CPU numerical tests; never fit a production calibration pool."""
import contextlib
import copy
from dataclasses import replace
import io
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.experiments import baselines, pbr


def records(n=20):
    return [{"id": i, "choices": list("ABCDE"), "gold": int(i % 3 == 0),
             "model_prediction": {"final_answer": "A", "confidence": i / max(1, n - 1)}}
            for i in range(n)]


def small_config(**kwargs):
    return replace(pbr.PBRConfig(n_inducing=6, steps=5, batch_size=7, mc_samples=12,
                                  alpha_grid=(.01, .5)), **kwargs)


class PBRNumericalTests(unittest.TestCase):
    def test_whitened_kl_against_distribution_implementation(self):
        dtype = torch.float64
        mean = torch.tensor([.2, -.4], dtype=dtype)
        chol = torch.tensor([[1.1, 0.], [.3, .8]], dtype=dtype)
        actual = pbr.gaussian_kl(mean, chol)
        posterior = torch.distributions.MultivariateNormal(mean, scale_tril=chol)
        prior = torch.distributions.MultivariateNormal(torch.zeros(2, dtype=dtype),
                                                      torch.eye(2, dtype=dtype))
        expected = torch.distributions.kl_divergence(posterior, prior)
        self.assertAlmostEqual(float(actual), float(expected), places=13)
        model = pbr.SparseBinaryGP(small_config())
        self.assertEqual(float(model.kl().detach()), 0.)

    def test_expected_brier_keeps_posterior_variance(self):
        probabilities = torch.tensor([[.1, .4], [.5, .6], [.9, .8]], dtype=torch.float64)
        labels = torch.tensor([0., 1.], dtype=torch.float64)
        mean_loss = 2 * (labels - probabilities.mean(0)).square().mean()
        variance = 2 * probabilities.var(0, unbiased=False).mean()
        self.assertAlmostEqual(float(pbr.expected_binary_brier(probabilities, labels)),
                               float(mean_loss + variance), places=14)
        self.assertGreater(float(variance), 0.)

    def test_kl_denominator_is_full_fold_not_batch_size(self):
        for batch_size in [1, 7, 20]:
            samples = torch.full((4, batch_size), .6, dtype=torch.float64)
            loss = pbr.expected_binary_brier(samples, torch.ones(batch_size))
            total = pbr.pac_bayes_objective(loss, torch.tensor(3.), .5, 100)
            self.assertAlmostEqual(float(total - loss), .5 / 100 * 3, places=8)

    def test_shared_gp_cross_covariance_and_symmetry(self):
        config = small_config()
        model = pbr.SparseBinaryGP(config)
        c = torch.tensor([.1, .3, .5, .7, .9], dtype=torch.float64)
        mean, variance = model.difference_moments(c)
        expected = 2 * config.kernel_variance * (1 - torch.exp(
            -.5 * ((2 * c - 1) / config.kernel_lengthscale).square()))
        np.testing.assert_allclose(variance.detach(), expected, atol=1e-12)
        self.assertEqual(float(variance[2].detach()), 0.)
        self.assertEqual(float(mean[2].detach()), 0.)
        samples = model.probability_samples(model.difference_features(c), 500,
                                            torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(samples[:, 2], torch.full((500,), .5, dtype=torch.float64)))
        pred = model.predictive_mean(c.numpy())
        np.testing.assert_allclose(pred + pred[::-1], 1., atol=1e-13)
        # Correlation from q(u), not just the original GP prior, is retained.
        with torch.no_grad():
            model.q_mean[0] = .3
            model.q_raw_tril[1, 0] = .4
        points = torch.cat([c[:2], 1 - c[:2]])
        a = model._features(points)
        lq = model.scale_tril()
        joint_cov = model.kernel(points, points) - a @ a.T + (a @ lq) @ (a @ lq).T
        _, actual = model.difference_moments(c[:2])
        expected = joint_cov.diag()[:2] + joint_cov.diag()[2:] - 2 * joint_cov[:2, 2:].diag()
        np.testing.assert_allclose(actual.detach(), expected.detach(), atol=1e-12)

    def test_endpoints_gradients_and_posterior_predictive_integral(self):
        config = small_config()
        model = pbr.SparseBinaryGP(config)
        c = torch.tensor([0., .25, .5, .75, 1.], dtype=torch.float64)
        samples = model.probability_samples(model.difference_features(c), 100,
                                            torch.Generator().manual_seed(42))
        loss = pbr.pac_bayes_objective(pbr.expected_binary_brier(samples, torch.ones(5)),
                                       model.kl(), .5, 5)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        probabilities = model.predictive_mean(c.numpy())
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        mean, variance = model.difference_moments(torch.tensor([.2], dtype=torch.float64))
        from scipy.integrate import quad
        from scipy.special import expit
        from scipy.stats import norm
        expected = quad(lambda x: expit(float(mean.detach()) + np.sqrt(float(variance.detach())) * x)
                        * norm.pdf(x), -10, 10, epsabs=1e-12)[0]
        self.assertAlmostEqual(model.predictive_mean([.2])[0], expected, places=10)
        self.assertNotAlmostEqual(expected, expit(float(mean.detach())), places=4)

    def test_fit_cv_refit_reproducibility_and_rng_isolation(self):
        data = records()
        config = small_config()
        original = copy.deepcopy(data)
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        first = pbr.fit_pbr(data, config=config)
        second = pbr.fit_pbr(data, config=config)
        self.assertEqual(first["best_alpha"], second["best_alpha"])
        np.testing.assert_array_equal(first["gp"].predictive_mean([.1, .4, .9]),
                                      second["gp"].predictive_mean([.1, .4, .9]))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], current_numpy[1])
        self.assertEqual(numpy_state[2:], current_numpy[2:])
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(data, original)
        self.assertEqual(first["metadata"]["pbr_calib_n"], 20)
        self.assertEqual(first["metadata"]["final_training"]["n_fit"], 20)
        validation_ids = []
        for fold in first["metadata"]["cv_folds"]:
            self.assertEqual(fold["n_fit"], 16)
            validation_ids.extend(fold["validation_ids"])
        self.assertEqual(sorted(validation_ids), list(range(20)))
        for score in first["metadata"]["cv_scores"]:
            for state in score["training"]:
                self.assertEqual(state["kl_coefficient"], score["alpha"] / 16)
        for a, b in zip(first["metadata"]["cv_scores"], second["metadata"]["cv_scores"]):
            self.assertEqual(a["fold_brier"], b["fold_brier"])
        # Unknown is retained; correctness uses the same pre-existing definition.
        data[0]["model_prediction"]["final_answer"] = "Unknown"
        fitted = pbr.fit_pbr(data, config=small_config(steps=1, alpha_grid=(.1,)))
        self.assertEqual(fitted["metadata"]["calibration_ids"], list(range(20)))

    def test_apply_needs_no_gold_or_raw_probabilities_and_artifact_roundtrip(self):
        data = records()
        with tempfile.TemporaryDirectory() as directory:
            model = baselines.fit_pbr(data, config=small_config(), artifact_dir=directory)
            evaluation = copy.deepcopy(data[:4])
            for r in evaluation:
                del r["gold"]
            snapshot = copy.deepcopy(evaluation)
            output = baselines.apply_pbr(model, evaluation)
            self.assertEqual(evaluation, snapshot)
            self.assertEqual([r["id"] for r in output], [r["id"] for r in evaluation])
            self.assertEqual([r["model_prediction"]["final_answer"] for r in output], ["A"] * 4)
            loaded = pbr.load_pbr(directory)
            self.assertEqual(output, baselines.apply_pbr(loaded, evaluation))
            with self.assertRaises(FileExistsError):
                pbr.save_pbr(model, directory)
            with self.assertRaises(ValueError):
                baselines.apply_pbr({"theta": np.array([.5])}, evaluation)

    def test_missing_full_pool_rejected_before_training(self):
        with patch.object(pbr, "_fit") as fit:
            with self.assertRaisesRegex(ValueError, "9741"):
                pbr.fit_pbr(records(), expected_n=9741, config=small_config())
            fit.assert_not_called()

    def test_both_comparison_entrypoints_keep_pbr_full_and_gp_1500(self):
        from src.experiments import experiment_all_baselines, experiment_label_free_baselines
        calibration = records(9741)
        evaluation = records(3)
        # Mock fitting only: inspect all actual IDs passed by each real entrypoint.
        for experiment in (experiment_all_baselines, experiment_label_free_baselines):
            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(experiment, "load_pair_data",
                    return_value=(calibration, [], evaluation, evaluation)))
                stack.enter_context(patch.object(experiment, "load_jsonl", return_value=[]))
                for name in ("apply_bct", "apply_ensemble", "apply_conf_avg"):
                    stack.enter_context(patch.object(experiment, name, return_value=evaluation))
                for name in ("fit_temperature_scaling", "fit_histogram_binning", "fit_isotonic_regression"):
                    if hasattr(experiment, name):
                        stack.enter_context(patch.object(experiment, name, return_value=1.))
                for name in ("apply_temperature_scaling", "apply_histogram_binning", "apply_isotonic_regression", "apply_gpcalib", "apply_pbr"):
                    if hasattr(experiment, name):
                        stack.enter_context(patch.object(experiment, name, return_value=evaluation))
                gp_fit = stack.enter_context(patch.object(experiment, "fit_gpcalib"))
                pbr_fit = stack.enter_context(patch.object(experiment, "fit_pbr", return_value={"best_alpha": .1}))
                stack.enter_context(patch.object(experiment, "pbr_result_metadata",
                                   return_value={"pbr_calib_n": 9741, "pbr_alpha": .1}))
                output, _ = experiment.run_pair("CommonSenseQA", "instruct", "base", "/unused")
                self.assertEqual(len(gp_fit.call_args.args[0]), 1500)
                self.assertEqual([r["id"] for r in gp_fit.call_args.args[0]],
                                 [r["id"] for r in baselines._subsample_for_gp(calibration, 1500)])
                self.assertIs(pbr_fit.call_args.args[0], calibration)
                self.assertEqual(pbr_fit.call_args.kwargs["expected_n"], 9741)
                self.assertEqual(output["PBR"]["pbr_calib_n"], 9741)
                self.assertNotIn("gp_calib_n", output["PBR"])
                self.assertEqual(output["GPcalib"]["gp_calib_n"], 1500)


if __name__ == "__main__":
    unittest.main()
