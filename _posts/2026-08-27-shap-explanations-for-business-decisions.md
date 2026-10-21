---
title: "SHAP Explained: How to Show Why a Model Made a Decision"
date: 2026-08-27T10:00:00+02:00
excerpt: "A model that cannot explain itself is hard to trust, hard to debug and hard to defend. How SHAP values split a prediction fairly between features, the game theory behind them worked through by hand, a modular Python toolkit that turns them into plain-language reasons, and the pitfalls that make explanations misleading."
categories:
  - Machine Learning
  - Explainable AI
tags:
  - SHAP
  - Explainability
  - Shapley Values
  - LightGBM
  - Model Governance
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "magnifying-glass-chart"
---

Sooner or later, every model I have deployed has been asked the same question by someone outside the
data team: "Why did it say that?" A sales manager wants to know why a loyal-looking customer is
flagged as likely to leave. A planner wants to know why next week's forecast is so high. An auditor
wants to know which inputs drive an automated decision. SHAP is the method I use most to answer those
questions, because it gives an explanation that adds up exactly to the prediction and treats features
fairly. In this post I explain the idea, work through the math by hand, show how I package it in
Python, and point out where explanations can mislead.

## The business problem

Machine learning models increasingly influence real decisions: who gets a retention offer, how much
stock to order, which ticket goes to a senior specialist. When people cannot see why a model behaves
as it does, four problems follow:

- **Low adoption.** Teams ignore recommendations they do not understand and fall back on their own
  judgement.
- **Hidden errors.** A model that relies on a leaked or broken feature can look accurate in testing
  and fail in production. Without explanations, nobody notices what it is actually relying on.
- **Hard conversations.** Customers, managers and auditors ask for reasons. "The model said so" is not
  an acceptable answer.
- **Regulatory pressure.** Data protection law and the EU AI Act both push towards transparency and
  meaningful information about automated decisions, especially where people are affected.

Good explanations turn a black box into a tool that people can question, correct and trust.

## The idea in plain English

Think of a prediction as a team result, and the features as the players. The model starts from a
**baseline**: the average prediction over a reference dataset. Each feature then pushes the
prediction up or down from that baseline. A SHAP value is the size of one feature's push.

The question is how to share the credit fairly when features interact. SHAP borrows the answer from
game theory: imagine adding the features one at a time in every possible order, and record how much
the prediction changes when each feature joins. A feature's SHAP value is its *average* contribution
over all those orders. That way, no feature gets extra credit just because of where it happened to be
added.

Two properties make this practical:

1. **It adds up.** Baseline plus all SHAP values equals the model's prediction for that case. Nothing
   is left unexplained.
2. **It works locally and globally.** For one customer you get "these three things raised the risk".
   Averaged over many customers you get "these are the features the model relies on most".

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Trained model</div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Background data<small>defines the baseline</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">SHAP explainer<small>one value per feature per case</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-branch">
      <div class="dg-node">Local view<small>top reasons for one case</small></div>
      <div class="dg-node">Global view<small>importance across all cases</small></div>
    </div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-ok">Reports, reviews and monitoring</div>
  </div>
  <p class="dg-caption">From a trained model to explanations that people can act on.</p>
</div>

## How it works: the math, step by step

### The Shapley value

Let $$F$$ be the set of all $$M$$ features, and let $$v(S)$$ be the model's expected output when only
the features in subset $$S$$ are known (the "value" of that coalition of features). The Shapley value
of feature $$i$$ is

$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}}
\frac{\lvert S \rvert! \, (M - \lvert S \rvert - 1)!}{M!}
\Big[ v(S \cup \{i\}) - v(S) \Big]
$$

Line by line:

- The sum runs over every subset $$S$$ of the *other* features.
- $$v(S \cup \{i\}) - v(S)$$ is the marginal contribution: how much the prediction changes when
  feature $$i$$ joins the features already in $$S$$.
- The fraction is the share of all $$M!$$ orderings in which exactly the features in $$S$$ come
  before $$i$$. There are $$\lvert S \rvert!$$ ways to order the features before it and
  $$(M - \lvert S \rvert - 1)!$$ ways to order those after it.
- So $$\phi_i$$ is the average marginal contribution of feature $$i$$ over all possible orders of
  arrival.

### What "only some features are known" means

For a specific case $$x$$, the value function is usually defined as an expectation over the features
that are not in $$S$$, drawn from background data. The **interventional** version is

$$
v(S) = E_{X_{\bar{S}}}\big[ f(x_S, X_{\bar{S}}) \big]
$$

In plain English: keep the case's own values for the features in $$S$$, fill in the others with
values from the background data, and average the model's output. The empty set gives the baseline
$$v(\emptyset) = E[f(X)]$$, the average prediction.

### The guarantees

Shapley values are the only attribution that satisfies all of these at once:

- **Efficiency.** The contributions add up to the prediction minus the baseline:

$$
\sum_{i=1}^{M} \phi_i = f(x) - E[f(X)]
$$

- **Symmetry.** Two features that always contribute the same get the same value.
- **Dummy.** A feature that never changes the output gets zero.
- **Additivity.** For a model that is a sum of two models (like a tree ensemble), the explanation is
  the sum of the two explanations.

Efficiency is the property business users care about most. It means the explanation is complete:
"baseline 10 percent, plus these pushes, equals 40 percent".

### A small example by hand

Take a model with two features and an interaction:

$$
f(x_1, x_2) = 10 + 4x_1 + 2x_2 + 3x_1 x_2
$$

Use a single background point $$(0, 0)$$, and explain the case $$x = (1, 1)$$. The four coalition
values are

$$
v(\emptyset) = f(0,0) = 10, \quad v(\{1\}) = f(1,0) = 14, \quad v(\{2\}) = f(0,1) = 12, \quad v(\{1,2\}) = f(1,1) = 19
$$

With two features there are two orders. Feature 1 either arrives first (contribution
$$14 - 10 = 4$$) or second (contribution $$19 - 12 = 7$$). Its Shapley value is the average:

$$
\phi_1 = \tfrac{1}{2}(4) + \tfrac{1}{2}(7) = 5.5,
\qquad
\phi_2 = \tfrac{1}{2}(12 - 10) + \tfrac{1}{2}(19 - 14) = 3.5
$$

Check efficiency: $$5.5 + 3.5 = 9 = 19 - 10$$. The main effects (4 and 2) went to their own features,
and the interaction term (3) was split evenly between the two features that create it. That is the
fairness the method promises.

### Making it fast

The formula needs $$2^{M}$$ coalitions, which is impossible for 50 features. Two algorithms make it
practical:

- **TreeSHAP** computes exact values for tree models (gradient boosting, random forests) in time
  proportional to the number of trees, leaves and depth, roughly $$O(T L D^2)$$, instead of
  exponential time. This is what I use for almost all tabular models.
- **KernelSHAP** works with any model by sampling coalitions and fitting a weighted linear
  regression. It is slower and approximate, so I use it only when there is no faster option.

### Units: log-odds versus probability

For classifiers, tree models usually explain the **log-odds** (the raw score before the sigmoid),
because that is where contributions add up. A probability is

$$
p = \frac{1}{1 + e^{-(\phi_0 + \sum_i \phi_i)}}
$$

where $$\phi_0$$ is the baseline log-odds. In plain English: add the baseline and the pushes on the
log-odds scale, then convert once to a probability. The pushes do not add up on the probability
scale, so I always show the final probability next to the reasons, not percentage contributions per
feature.

## Implementation in Python

### Computing explanations

```python
# explain/shap_tools.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


@dataclass(frozen=True)
class ExplainConfig:
    background_size: int = 500
    top_k: int = 3
    seed: int = 7


def build_explainer(model, X_reference: pd.DataFrame, cfg: ExplainConfig = ExplainConfig()):
    """Interventional TreeSHAP with an explicit, sampled background set."""
    background = X_reference.sample(min(cfg.background_size, len(X_reference)), random_state=cfg.seed)
    return shap.TreeExplainer(model, data=background, feature_perturbation="interventional")


def explain(explainer, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return SHAP values (rows x features) and the baseline, for the positive class."""
    exp = explainer(X)
    values = exp.values
    base = np.atleast_1d(exp.base_values)
    if values.ndim == 3:            # some models return one slice per class
        values, base = values[:, :, 1], base[:, 1] if base.ndim > 1 else base
    return values, float(np.ravel(base)[0])


def global_importance(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    imp = np.abs(values).mean(axis=0)
    return (pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))
```

### Turning numbers into reasons people can read

```python
# explain/reasons.py
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = {
    "days_since_login": "Days since last login",
    "support_tickets_90d": "Support tickets in the last 90 days",
    "seat_utilisation": "Share of paid seats in use",
    "months_to_renewal": "Months until renewal",
}


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def top_reasons(row: pd.Series, shap_row: np.ndarray, base: float, k: int = 3) -> dict:
    order = np.argsort(-np.abs(shap_row))[:k]
    reasons = []
    for i in order:
        name = row.index[i]
        direction = "raised" if shap_row[i] > 0 else "lowered"
        reasons.append(f"{LABELS.get(name, name)} ({row.iloc[i]}) {direction} the risk")
    return {
        "probability": round(float(sigmoid(base + shap_row.sum())), 3),
        "baseline_probability": round(float(sigmoid(base)), 3),
        "reasons": reasons,
    }
```

### Using it

```python
from lightgbm import LGBMClassifier

from explain.reasons import top_reasons
from explain.shap_tools import build_explainer, explain, global_importance

model = LGBMClassifier(n_estimators=400, learning_rate=0.05).fit(X_train, y_train)

explainer = build_explainer(model, X_train)
values, base = explain(explainer, X_score)

print(global_importance(values, list(X_score.columns)).head(10))

i = X_score.index.get_loc(account_id)
print(top_reasons(X_score.iloc[i], values[i], base))

# for analysts: shap.plots.beeswarm(explainer(X_score)) and shap.plots.waterfall(explainer(X_score)[i])
```

### Checking the explanation is complete

```python
import numpy as np


def check_additivity(model, X, values: np.ndarray, base: float, tol: float = 1e-3) -> float:
    """Largest gap between model log-odds and baseline + SHAP sum."""
    raw = model.predict(X, raw_score=True)
    gap = np.abs(raw - (base + values.sum(axis=1)))
    if gap.max() > tol:
        raise AssertionError(f"SHAP values do not add up (max gap {gap.max():.4f})")
    return float(gap.max())
```

I run this check in the test suite. If it ever fails, either the explainer is set up for the wrong
output scale or the model and explainer have drifted apart.

## A worked example: explaining churn risk to a customer success team

A software company sells annual subscriptions to about 4,000 business customers. A churn model scores
every account each week, and the customer success team calls the highest-risk accounts. At first the
team ignores the list, because it does not say *why* an account is risky.

With SHAP, one account's explanation looks like this, on the log-odds scale:

- Baseline log-odds: $$-2.2$$, which is a probability of $$1 / (1 + e^{2.2}) \approx 0.10$$.
- Days since last login (42): $$+1.1$$
- Support tickets in the last 90 days (7): $$+0.6$$
- Months until renewal (2): $$+0.4$$
- Share of paid seats in use (0.85): $$-0.3$$

The total is $$-2.2 + 1.1 + 0.6 + 0.4 - 0.3 = -0.4$$, so the predicted probability is

$$
p = \frac{1}{1 + e^{0.4}} \approx 0.40
$$

The account manager now sees: "40 percent risk, up from a typical 10 percent, mainly because nobody
has logged in for six weeks, there were seven support tickets recently, and renewal is in two months.
Seat usage is still healthy." That is a reason to call, and a script for the call: ask about the
support issues and re-engage the inactive users before the renewal discussion.

Globally, the mean absolute SHAP values also revealed something useful: a field called
`account_status_code` was the third most important feature. On inspection, one of its values was only
set *after* a customer had given notice. It was leaking the outcome. Removing it lowered offline
accuracy slightly but made the model honest, which no accuracy metric would have shown.

## Pitfalls and when not to use it

- **Reading SHAP as cause and effect.** SHAP explains the *model*, not the world. "Support tickets
  raised the risk" means the model uses that signal, not that reducing tickets will reduce churn.
  Causal questions need experiments or causal methods.
- **Correlated features.** When two features carry the same information, credit is split between
  them in ways that can look arbitrary. Group correlated features, or explain at the group level.
- **A poor background set.** The baseline depends on the background data. Use a representative
  sample of the population you score, not a convenient subset.
- **Mixing scales.** Contributions add on the log-odds scale, not the probability scale. Show the
  final probability and the direction of each reason, not "feature X added 12 percent".
- **Explaining an unstable model.** If the model changes a lot between retrains, so will the
  explanations. Stabilise the model first.
- **When a simple model is enough.** If a logistic regression with a few features performs as well,
  its coefficients may be the clearest explanation of all.

## Measuring success in production

- Adoption: share of model recommendations acted on, before and after adding explanations.
- Issues found through explanations (leaks, broken features, unwanted reliance).
- Stability of global importance between model versions.
- Time needed to answer "why did the model say this?" from a stakeholder or auditor.

## Checklist

- Choose a representative background set and store it with the model version.
- Use TreeSHAP for tree models and know which output scale you are explaining.
- Test that baseline plus SHAP values equals the model output.
- Translate the top reasons into plain-language labels for business users.
- Review global importance for leaks and unwanted features before every release.
- Never present SHAP values as causal effects.

## Final thought

Explanations are not decoration on top of a model. In my experience they are one of the best tools
for finding mistakes before they reach the business, and the fastest way to earn the trust of the
people who have to act on the predictions. If you want explainability built into your models and
reports, see my [forecasting, optimisation and decision analytics service](/services/#forecasting-decision-analytics) or
[contact me](/contact/).
