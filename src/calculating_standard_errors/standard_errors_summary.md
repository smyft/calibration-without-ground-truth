# Standard Errors for Evaluation Metrics on a Fixed Test Set

This note summarizes closed-form (non-bootstrap, single-evaluation) standard error formulas for four evaluation metrics: **Accuracy**, **Brier Score**, **Confidence Loss**, and **Expected Calibration Error (ECE)**. All formulas treat the test set as i.i.d. samples from the underlying data distribution and quantify uncertainty in the metric *as an estimate of its population value*. No model retraining or resampling is required.

## Notation

- $n$: test set size
- $y_i \in \{1, \dots, K\}$: ground-truth label of example $i$
- $\hat{y}_i$: predicted class for example $i$
- $\hat{c}_i \in [0, 1]$: predicted confidence (probability of the predicted class) for example $i$
- $z_i = \mathbf{1}\{\hat{y}_i = y_i\}$: correctness indicator for example $i$

---

## 1. Accuracy

Accuracy is the mean of the correctness indicator:

$$
\widehat{\mathrm{Acc}} = \frac{1}{n} \sum_{i=1}^n z_i.
$$

Since each $z_i \in \{0, 1\}$ is a Bernoulli trial, the standard error follows the binomial formula:

$$
\widehat{\mathrm{SE}}(\widehat{\mathrm{Acc}}) = \sqrt{\frac{\widehat{\mathrm{Acc}}(1 - \widehat{\mathrm{Acc}})}{n}}.
$$

A 95% normal CI is $\widehat{\mathrm{Acc}} \pm 1.96 \cdot \widehat{\mathrm{SE}}(\widehat{\mathrm{Acc}})$.

---

## 2. Brier Score

Our setting provides only a (final answer, confidence) pair per example rather than a full probability distribution. The per-example Brier loss is therefore the binary form:

$$
b_i = (\hat{c}_i - z_i)^2.
$$

The Brier Score is the sample mean:

$$
\widehat{\mathrm{BS}} = \frac{1}{n} \sum_{i=1}^n b_i.
$$

Because $\widehat{\mathrm{BS}}$ is a mean of i.i.d. per-example values, its standard error is the standard error of the mean:

$$
\widehat{\mathrm{SE}}(\widehat{\mathrm{BS}}) = \frac{s_b}{\sqrt{n}}, \qquad
s_b^2 = \frac{1}{n - 1} \sum_{i=1}^n (b_i - \widehat{\mathrm{BS}})^2.
$$

A 95% normal CI is $\widehat{\mathrm{BS}} \pm 1.96 \cdot \widehat{\mathrm{SE}}(\widehat{\mathrm{BS}})$.

---

## 3. Confidence Loss (Brier Score − Accuracy)

Define the per-example confidence loss contribution:

$$
\ell_i = b_i - z_i.
$$

The Confidence Loss is again a sample mean:

$$
\widehat{\mathrm{CL}} = \frac{1}{n} \sum_{i=1}^n \ell_i = \widehat{\mathrm{BS}} - \widehat{\mathrm{Acc}}.
$$

Compute the SE directly from the per-example values $\ell_i$:

$$
\widehat{\mathrm{SE}}(\widehat{\mathrm{CL}}) = \frac{s_\ell}{\sqrt{n}}, \qquad
s_\ell^2 = \frac{1}{n - 1} \sum_{i=1}^n (\ell_i - \widehat{\mathrm{CL}})^2.
$$

> **Important:** Do **not** combine $\widehat{\mathrm{SE}}(\widehat{\mathrm{BS}})$ and $\widehat{\mathrm{SE}}(\widehat{\mathrm{Acc}})$ via $\sqrt{\widehat{\mathrm{SE}}(\widehat{\mathrm{BS}})^2 + \widehat{\mathrm{SE}}(\widehat{\mathrm{Acc}})^2}$. The Brier Score and Accuracy are computed on the same examples and are correlated, so this naive combination is incorrect. Computing the SE from the per-example differences $\ell_i$ correctly accounts for the correlation.

A 95% normal CI is $\widehat{\mathrm{CL}} \pm 1.96 \cdot \widehat{\mathrm{SE}}(\widehat{\mathrm{CL}})$.

---

## 4. Expected Calibration Error (ECE)

ECE bins predictions by confidence and compares per-bin accuracy to per-bin mean confidence. Let $B_1, \dots, B_M$ be the bins with sizes $n_m = |B_m|$, and define the per-example gap contribution:

$$
d_i = z_i - \hat{c}_i.
$$

Per-bin mean gap:

$$
\widehat{\Delta}_m = \frac{1}{n_m} \sum_{i \in B_m} d_i.
$$

ECE estimator:

$$
\widehat{\mathrm{ECE}} = \sum_{m=1}^M \frac{n_m}{n} \, \big| \widehat{\Delta}_m \big|.
$$

### Delta-method standard error

Because each example contributes to exactly one bin, the per-bin means $\widehat{\Delta}_m$ are independent. Estimate the variance of each per-bin mean by the standard SE-of-the-mean:

$$
\widehat{\mathrm{Var}}(\widehat{\Delta}_m) = \frac{s_m^2}{n_m}, \qquad
s_m^2 = \frac{1}{n_m - 1} \sum_{i \in B_m} (d_i - \widehat{\Delta}_m)^2.
$$

Apply the delta method to the absolute value (whose derivative is $\pm 1$ wherever defined) and combine across bins:

$$
\widehat{\mathrm{SE}}(\widehat{\mathrm{ECE}}) = \frac{1}{n} \sqrt{\sum_{m=1}^M n_m^2 \cdot \frac{s_m^2}{n_m}} = \frac{1}{n} \sqrt{\sum_{m=1}^M n_m \, s_m^2}.
$$

A 95% normal CI is $\widehat{\mathrm{ECE}} \pm 1.96 \cdot \widehat{\mathrm{SE}}(\widehat{\mathrm{ECE}})$.

### Caveats

The delta-method approximation treats bin assignments as fixed and uses the linearization of $|x|$, whose derivative is undefined at zero. The CI is therefore most reliable when:

- Each bin contains enough samples (a rule of thumb: $n_m \gtrsim 30$).
- The estimated bin gaps $|\widehat{\Delta}_m|$ are not extremely close to zero relative to their standard errors.

When true bin gaps are near zero, $|\widehat{\Delta}_m|$ behaves like a folded normal rather than a normal random variable, so the delta-method SE may underestimate variability and the resulting CI can be slightly too narrow. If equal-mass binning is used (bin edges depend on the predicted confidences), the bins are technically random and there is an additional small variance term being ignored; this is standard practice but worth acknowledging.

---

## Summary Table

| Metric | Estimator | Standard Error |
|---|---|---|
| Accuracy | $\frac{1}{n} \sum_i z_i$ | $\sqrt{\widehat{\mathrm{Acc}}(1 - \widehat{\mathrm{Acc}})/n}$ |
| Brier Score | $\frac{1}{n} \sum_i (\hat{c}_i - z_i)^2$ | $s_b / \sqrt{n}$ |
| Confidence Loss | $\frac{1}{n} \sum_i ((\hat{c}_i - z_i)^2 - z_i)$ | $s_\ell / \sqrt{n}$ |
| ECE | $\sum_m \frac{n_m}{n} \lvert \widehat{\Delta}_m \rvert$ | $\frac{1}{n} \sqrt{\sum_m n_m \, s_m^2}$ |

All four standard errors are computed from a single evaluation pass on the fixed test set — no retraining and no resampling required.
