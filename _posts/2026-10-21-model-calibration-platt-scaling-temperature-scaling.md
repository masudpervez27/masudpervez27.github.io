---
title: "Can You Trust a 90% Prediction? Calibrating Models with Platt Scaling and Temperature Scaling"
date: 2026-10-21T00:05:00+02:00
excerpt: "Most models are confident in the wrong way: when they say 90 percent, they are right far less often. How to measure that gap, fix it with Platt scaling and temperature scaling, and why calibrated probabilities change real decisions on thresholds, capacity and automation."
categories:
  - Machine Learning
  - Decision Science
tags:
  - Probability Calibration
  - Platt Scaling
  - Temperature Scaling
  - Model Evaluation
  - Uncertainty
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "sliders"
---

When a model says "90 percent", I want that to mean something: out of a hundred cases it scores at
90 percent, about ninety should turn out positive. Surprisingly often that is not true. Modern neural
networks are typically overconfident, boosted trees and margin-based models produce scores that are
not probabilities at all, and models trained with class weights or resampling are shifted on purpose.
The ranking may be excellent while the numbers themselves are wrong. This post explains how I measure
that problem, the two calibration methods I use most (Platt scaling and temperature scaling), the
math behind them line by line, and the practical difference calibrated probabilities make to business
decisions.

## The business problem

A probability is only useful if you can act on it as a probability. In production, models rarely
just rank; their outputs feed decisions with money attached:

- **Cost-based thresholds.** "Act if the probability is above the ratio of action cost to loss" only
  works if the probability is honest (see my post on
  [cost-sensitive decision thresholds](/posts/2026-01-15-cost-sensitive-classification-decision-thresholds/)).
- **Automation rules.** "Auto-route the ticket if confidence is above 0.9" quietly assumes that 0.9
  means 90 percent correct. If the model is overconfident, far more errors slip through than planned.
- **Expected values.** Revenue forecasts, capacity plans and risk budgets often sum probabilities
  across many cases. A systematic bias of a few percentage points becomes a large planning error.
- **Combining models.** When one model's output becomes another system's input, miscalibration
  spreads through everything downstream.

The worst part is that an uncalibrated model can look excellent on the usual metrics. AUC and
accuracy only care about ranking and the final class, so nobody notices until the decisions built on
the probabilities start to go wrong.

## The idea in plain English

Calibration is a small correction applied *after* the model is trained. You keep the model exactly as
it is, and learn a simple mapping from its raw output to an honest probability, using data the model
has not seen.

- **Platt scaling** fits a tiny logistic regression on top of the model's score: it can shift the
  scores and stretch or squeeze them. It is my default for binary problems, especially for tree
  models and margin-based models.
- **Temperature scaling** divides the model's logits by a single number, the temperature, before the
  softmax. A temperature above 1 softens overconfident predictions; below 1 sharpens timid ones. It
  never changes which class wins, so accuracy stays exactly the same. It is my default for neural
  networks and multi-class problems.

Both methods have only one or two parameters, so they need little data and are hard to overfit. That
simplicity is the reason they work so well in practice.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Trained model<small>unchanged</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Calibration set<small>data the model has not seen</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-branch">
      <div class="dg-node">Platt scaling<small>binary scores</small></div>
      <div class="dg-node">Temperature scaling<small>multi-class logits</small></div>
    </div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">Check reliability<small>ECE, Brier, log loss</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-ok">Honest probabilities<small>thresholds and automation</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-warn">Monitor and recalibrate</div>
  </div>
  <p class="dg-caption">Calibration is a small, separate step between the model and the decisions that use it.</p>
</div>

## How it works: the math, step by step

### What "calibrated" means

A model that outputs probability $$\hat{p}$$ for the positive class is calibrated if

$$
P\big(Y = 1 \mid \hat{p} = p\big) = p \quad \text{for every } p \in [0, 1]
$$

In plain English: among all cases where the model says $$p$$, the share that are actually positive
is $$p$$. For multi-class models, the usual version looks at the predicted class and its confidence:
among predictions made with confidence 0.8, about 80 percent should be correct.

### Measuring it: reliability and the expected calibration error

Split predictions into $$B$$ bins by confidence (for example 0 to 0.1, 0.1 to 0.2, and so on). In
each bin $$b$$ with $$n_b$$ predictions, compare the average confidence $$\text{conf}_b$$ with the
observed accuracy $$\text{acc}_b$$. The **expected calibration error** is

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \, \big\lvert \text{acc}_b - \text{conf}_b \big\rvert
$$

Line by line:

- $$\lvert \text{acc}_b - \text{conf}_b \rvert$$ is the gap between what the model claimed and what
  happened in that bin.
- $$n_b / N$$ weights each bin by how many predictions fall into it, so busy bins count more.
- The sum is the average gap, in probability units. An ECE of 0.09 means the model is off by 9
  percentage points on average.

Plotting $$\text{acc}_b$$ against $$\text{conf}_b$$ gives the **reliability diagram**. A calibrated
model lies on the diagonal; an overconfident model lies below it.

### Proper scoring rules: log loss and Brier score

ECE depends on the binning, so I always report it together with two proper scoring rules, which are
minimised only by the true probabilities:

$$
\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \big[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \big]
\qquad
\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2
$$

The Brier score has a useful approximate decomposition (Murphy) with $$\bar{y}$$ the overall
positive rate:

$$
\text{Brier} \approx
\underbrace{\frac{1}{N}\sum_{b} n_b (\text{conf}_b - \text{acc}_b)^2}_{\text{reliability}}
- \underbrace{\frac{1}{N}\sum_{b} n_b (\text{acc}_b - \bar{y})^2}_{\text{resolution}}
+ \underbrace{\bar{y}(1 - \bar{y})}_{\text{uncertainty}}
$$

Read it as: the error is made of miscalibration (which calibration can fix), minus the model's
ability to separate cases (which only a better model can improve), plus the inherent difficulty of
the problem (which nobody can change). Calibration attacks the first term without touching the
second.

### Platt scaling

Let $$s$$ be the model's raw score for a case: a log-odds, a margin, or any number where larger means
"more likely positive". Platt scaling models the probability as

$$
\hat{p} = \sigma(A s + B) = \frac{1}{1 + e^{-(A s + B)}}
$$

In plain English: stretch or squeeze the score by $$A$$, shift it by $$B$$, and pass it through the
sigmoid. If the raw scores are overconfident, the fitted $$A$$ will be below 1. If the model was
trained on a rebalanced dataset, $$B$$ moves the probabilities back to the real base rate.

$$A$$ and $$B$$ are fitted by minimising the log loss on the calibration set. Platt also suggested
slightly softened targets to avoid overfitting with few examples. With $$N_+$$ positives and $$N_-$$
negatives:

$$
t_i =
\begin{cases}
\dfrac{N_+ + 1}{N_+ + 2} & \text{if } y_i = 1 \\[2ex]
\dfrac{1}{N_- + 2} & \text{if } y_i = 0
\end{cases}
$$

Read it as: never ask the calibrator to predict exactly 0 or 1, because a finite sample can never
justify that much certainty. The loss and its gradients are

$$
L(A, B) = -\sum_{i} \big[ t_i \ln \hat{p}_i + (1 - t_i) \ln(1 - \hat{p}_i) \big],
\qquad
\frac{\partial L}{\partial A} = \sum_i (\hat{p}_i - t_i)\, s_i,
\qquad
\frac{\partial L}{\partial B} = \sum_i (\hat{p}_i - t_i)
$$

In plain English: the parameters move until predicted and target probabilities agree on average
(the $$B$$ gradient) and agree in how they vary with the score (the $$A$$ gradient). This is an
ordinary two-parameter logistic regression, so it converges quickly and reliably.

### Temperature scaling

For a multi-class model with logits $$z_1, \dots, z_K$$ (the values before the softmax), temperature
scaling divides all logits by one positive number $$T$$:

$$
\hat{p}_k = \frac{e^{z_k / T}}{\sum_{j=1}^{K} e^{z_j / T}}
$$

Line by line:

- With $$T = 1$$ nothing changes.
- With $$T > 1$$ the logits move closer together, so the probabilities become less extreme. This
  fixes overconfidence, the typical problem of deep networks.
- With $$T < 1$$ the logits spread apart and probabilities become sharper.
- Dividing every logit by the same positive number never changes their order, so the predicted class
  and therefore the accuracy are exactly the same as before.

$$T$$ is fitted by minimising the negative log-likelihood on the calibration set:

$$
T^{*} = \arg\min_{T > 0} \; -\frac{1}{N} \sum_{i=1}^{N} \ln \hat{p}_{i, y_i}(T)
$$

In plain English: choose the temperature that makes the true classes as probable as possible on data
the model has not seen. It is a one-dimensional search, so it is fast and stable.

### How the two methods relate

For a binary model with logit $$z$$, the positive-class probability with temperature is
$$\sigma(z / T)$$. That is Platt scaling with $$A = 1/T$$ and $$B = 0$$. So temperature scaling is the
more restricted method: it can soften or sharpen, but it cannot shift. That restriction is its
strength for multi-class networks (one parameter, no change in accuracy), and its weakness when the
base rate has moved, which needs the shift $$B$$.

### Correcting for a changed base rate

If the share of positives changes between training and production, from $$\pi$$ to $$\pi'$$, and the
relationship between features and class is otherwise unchanged, Bayes' rule gives a simple correction
on the log-odds scale:

$$
\text{logit}(\hat{p}') = \text{logit}(\hat{p}) + \ln\frac{\pi'}{1 - \pi'} - \ln\frac{\pi}{1 - \pi}
$$

In plain English: add the change in prior log-odds to every prediction. This is exactly an update of
the Platt intercept $$B$$, which is why I keep a calibration layer separate from the model: it can be
refreshed in minutes when the base rate moves, without retraining.

## Implementation in Python

### Metrics

```python
# calibration/metrics.py
from __future__ import annotations

import numpy as np


def reliability_table(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Per-bin count, average confidence and observed accuracy."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"low": edges[b], "high": edges[b + 1], "n": int(m.sum()),
                         "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return rows


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    n = len(conf)
    return float(sum(r["n"] / n * abs(r["accuracy"] - r["confidence"])
                     for r in reliability_table(conf, correct, n_bins)))


def top_label(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Confidence of the predicted class and whether it was correct (multi-class)."""
    return probs.max(axis=1), (probs.argmax(axis=1) == y).astype(float)


def binary_scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "ece": expected_calibration_error(p, y.astype(float)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
    }
```

For binary problems I pass the positive-class probability and the label directly to the ECE
function, so each bin compares the predicted rate with the observed positive rate.

### Platt scaling

```python
# calibration/platt.py
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


class PlattScaler:
    """Fits p = sigmoid(A * score + B) with Platt's smoothed targets."""

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattScaler":
        s = np.asarray(scores, dtype=float)
        y = np.asarray(y, dtype=int)
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

        def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = w
            z = a * s + b
            loss = -np.sum(t * log_expit(z) + (1 - t) * log_expit(-z))
            g = expit(z) - t
            return float(loss), np.array([np.sum(g * s), np.sum(g)])

        res = minimize(loss_and_grad, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        self.a_, self.b_ = (float(v) for v in res.x)
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        return expit(self.a_ * np.asarray(scores, dtype=float) + self.b_)

    def shift_base_rate(self, old_rate: float, new_rate: float) -> None:
        """Adjust the intercept when the share of positives changes."""
        logit = lambda q: np.log(q / (1 - q))
        self.b_ += float(logit(new_rate) - logit(old_rate))
```

Using `log_expit` keeps the loss numerically stable for very large or very small scores, where
taking the log of a sigmoid directly would underflow.

### Temperature scaling

```python
# calibration/temperature.py
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


class TemperatureScaler:
    """Fits a single temperature T on logits by minimising the negative log-likelihood."""

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        z = np.asarray(logits, dtype=float)
        y = np.asarray(y, dtype=int)
        rows = np.arange(len(y))

        def nll(log_t: float) -> float:
            log_p = log_softmax(z / np.exp(log_t), axis=1)
            return float(-log_p[rows, y].mean())

        res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        self.temperature_ = float(np.exp(res.x))
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(logits, dtype=float) / self.temperature_, axis=1)
```

Searching over $$\ln T$$ instead of $$T$$ keeps the temperature positive and makes the search
symmetric between softening and sharpening.

### Getting raw scores and logits

- **scikit-learn:** `decision_function` for linear models and SVMs, or the log-odds of
  `predict_proba` for other classifiers.
- **LightGBM:** `model.predict(X, raw_score=True)` returns log-odds for binary and logits for
  multi-class.
- **Neural networks:** the outputs of the last layer before the softmax.

For binary problems I calibrate the log-odds with `PlattScaler`. For multi-class problems I pass the
logit matrix to `TemperatureScaler`.

### Choosing between methods with held-out data

```python
# calibration/select.py
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from calibration.metrics import binary_scores
from calibration.platt import PlattScaler


def compare_binary_calibrators(scores: np.ndarray, y: np.ndarray, seed: int = 0) -> dict[str, dict]:
    """Fit on one half of the calibration data, evaluate on the other half."""
    s_fit, s_eval, y_fit, y_eval = train_test_split(scores, y, test_size=0.5, stratify=y, random_state=seed)

    platt = PlattScaler().fit(s_fit, y_fit)
    iso = IsotonicRegression(out_of_bounds="clip").fit(s_fit, y_fit)

    return {
        "uncalibrated": binary_scores(expit(s_eval), y_eval),
        "platt": binary_scores(platt.predict_proba(s_eval), y_eval),
        "isotonic": binary_scores(iso.predict(s_eval), y_eval),
    }
```

I include isotonic regression as a comparison. It is more flexible than Platt scaling and can fix
odd-shaped distortions, but it needs much more data (as a rough guide, thousands of calibration
examples) and tends to overfit on small sets. If Platt and isotonic score about the same, I keep
Platt: two parameters are easier to monitor and explain.

## A worked example: automating ticket routing

A company receives about 12,000 support tickets a month. A text classifier assigns each ticket to one
of five teams. The plan: if the model's confidence is at least 0.9, route automatically; otherwise a
coordinator decides. Management accepted the plan on the assumption that automatically routed tickets
would be about 90 percent correct or better.

### Measuring the problem

On a held-out set of 1,000 tickets, the reliability table (simplified to three bins) looks like this:

| Confidence bin | Tickets | Average confidence | Accuracy |
|---|---|---|---|
| High | 500 | 0.95 | 0.82 |
| Medium | 300 | 0.75 | 0.70 |
| Low | 200 | 0.55 | 0.50 |

$$
\text{ECE} = \frac{500}{1000}(0.13) + \frac{300}{1000}(0.05) + \frac{200}{1000}(0.05) = 0.065 + 0.015 + 0.010 = 0.09
$$

The important number is in the first row: tickets the model is "95 percent sure" about are correct
only 82 percent of the time. Automatic routing would misroute nearly one in five of them, more than
three times the error rate the business signed off on.

### Fixing it

Temperature scaling on a separate calibration set finds $$T = 1.8$$. For one ticket with logits
$$(4.0, 2.0, 1.0)$$ over its top three teams (the other two are far lower and ignored here for
readability):

$$
T = 1: \quad \hat{p}_1 = \frac{e^{4}}{e^{4} + e^{2} + e^{1}} = \frac{54.60}{54.60 + 7.39 + 2.72} \approx 0.84
$$

$$
T = 1.8: \quad \hat{p}_1 = \frac{e^{2.22}}{e^{2.22} + e^{1.11} + e^{0.56}} = \frac{9.23}{9.23 + 3.04 + 1.74} \approx 0.66
$$

The predicted team is the same, but the confidence drops from 0.84 to 0.66, which is much closer to
how often such tickets are actually right. After calibration the ECE on the evaluation set falls from
about 0.09 to about 0.02, and the accuracy of the model is unchanged, as the math guarantees.

### What changes for the business

With honest probabilities, the 0.9 rule now means what it says. Fewer tickets clear the bar, so the
automation rate is lower than the original plan suggested, but the error rate among automated tickets
matches the agreed level. Instead of discovering misrouted tickets through customer complaints, the
team can make a deliberate trade-off: lower the threshold to automate more, and know in advance how
many errors that will cost.

### A binary case with Platt scaling

The same logic applies to a binary model. Suppose a gradient boosting model gives a case a raw
log-odds of $$s = 2.5$$, which it would report as $$\sigma(2.5) \approx 0.92$$. Platt scaling on a
calibration set finds $$A = 0.6$$ and $$B = -0.4$$, so

$$
\hat{p} = \sigma(0.6 \times 2.5 - 0.4) = \sigma(1.1) \approx 0.75
$$

If the business acts only above 0.8, the uncalibrated model would have triggered an action that the
calibrated probability does not justify. Across thousands of decisions a month, that difference is
the gap between the planned and the actual cost of the policy.

## Pitfalls and when not to use it

- **Calibrating on training data.** The model is overconfident on data it has seen. Always use a
  separate calibration set, or cross-validation.
- **Too little calibration data.** Platt and temperature scaling work with a few hundred examples;
  isotonic regression needs far more. With very few positives, prefer Platt with smoothed targets.
- **Expecting calibration to improve ranking.** Platt and temperature scaling are monotone, so AUC
  does not change. If the model cannot separate classes, calibration will not help.
- **One temperature for very different classes.** Temperature scaling applies the same correction to
  every class. If some classes are overconfident and others are not, consider vector or matrix
  scaling, or per-class Platt scaling.
- **Base-rate changes.** Calibration is tied to the population it was fitted on. When the share of
  positives changes, update the intercept or recalibrate.
- **Calibration drift.** Calibration degrades as data changes, often before accuracy does. Monitor
  ECE and the reliability diagram over time, alongside input drift (see my post on
  [population stability index](/posts/2026-08-13-population-stability-index-drift-detection/)).
- **Confidence stated in words by language models.** A language model saying "I am 90 percent sure"
  is not a calibrated probability. If you need confidence from an LLM, use token probabilities,
  agreement between repeated runs, or a separate calibrated classifier.

## Measuring success in production

- ECE, Brier score and log loss on recent labelled data, tracked monthly.
- Reliability diagram per model version, reviewed before each release.
- For automation rules: the actual error rate among automated cases versus the planned rate.
- For cost-based decisions: realised cost versus the cost the policy predicted.
- Time needed to recalibrate after a base-rate change (minutes, if the layer is separate).

## Checklist

- Report calibration (ECE, Brier, log loss), not only accuracy and AUC.
- Keep a calibration set that neither training nor model selection has seen.
- Use Platt scaling for binary scores and temperature scaling for multi-class logits.
- Compare with isotonic regression when you have plenty of calibration data.
- Store the calibrator with the model version and apply it in the same scoring service.
- Recalibrate, or adjust the intercept, when the base rate changes.
- Monitor calibration drift and the real error rate of every automation threshold.

## Final thought

Calibration is one of the cheapest improvements you can make to a model in production: a couple of
parameters, fitted in seconds, and no change to accuracy. What it changes is whether the numbers can
be trusted, and every threshold, automation rule and expected-value calculation depends on that. If
you want your models' probabilities to hold up when real decisions depend on them, see my
[real-time ML and decision systems service](/services/#real-time-ml-systems) or
[get in touch](/contact/).
