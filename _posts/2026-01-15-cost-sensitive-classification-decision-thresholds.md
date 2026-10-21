---
title: "Cost-Sensitive Classification: Choosing a Decision Threshold That Matches Your Business Costs"
date: 2026-01-15T10:00:00+01:00
excerpt: "A model that is 95 percent accurate can still lose money. How to turn predicted probabilities into decisions using the real cost of each kind of mistake, with the math, a reusable Python module and a predictive maintenance example."
redirect_from:
  - /posts/2026-01-15-cost-sensitive-machine-learning-in-real-time-fraud-detection/
categories:
  - Machine Learning
  - Decision Science
tags:
  - Classification
  - Decision Threshold
  - Probability Calibration
  - Cost-Sensitive Learning
  - scikit-learn
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "balance-scale"
---

The most common mistake I see in deployed classification models is not in the model. It is in the
single line of code that turns a probability into a decision: `prediction = probability > 0.5`. That
0.5 is almost never the right number, because it quietly assumes that every kind of mistake costs the
same. In real businesses they rarely do. This post shows how to pick the threshold from the costs
themselves, explains the math line by line, and gives you a small Python module you can drop into any
project.

## The business problem

Most classification models exist to trigger an action:

- Send a technician to inspect a machine that might fail.
- Offer a retention discount to a customer who might cancel.
- Route an incoming support ticket to a senior specialist.
- Send a product batch for manual quality inspection.

Each action has a cost, and each missed case has a different cost. Inspecting a healthy machine
wastes a few hours of a technician's time. Missing a machine that then breaks down can stop a
production line for a day. If you treat those two mistakes as equal, the model will be tuned for the
wrong goal, and the business will either spend too much on unnecessary actions or lose too much on
missed cases.

The metric that matters to the business is not accuracy or AUC. It is **expected cost** (or expected
profit) per decision. Once you frame the problem that way, the right threshold follows directly.

## The idea in plain English

A good classifier gives you a probability: "this machine has a 4 percent chance of failing this
week". The model's job ends there. Your job is to decide what to do with that 4 percent.

The question to ask is simple: *is acting on this case cheaper, on average, than not acting?*

- If you act, you pay the action cost, whether or not the failure would have happened.
- If you do not act, you pay the failure cost, but only if the failure actually happens.

When the failure cost is 30 times the action cost, acting becomes worthwhile even at quite low
probabilities. The threshold should therefore be set by the ratio of costs, not by the convention of
0.5.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Model<small>raw scores</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Calibration<small>scores become probabilities</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Cost matrix<small>agreed with the business</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">Decision rule<small>cost threshold or top-K by saving</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-branch">
      <div class="dg-node dg-ok">Act</div>
      <div class="dg-node">Do nothing</div>
    </div>
  </div>
  <p class="dg-note">A small random hold-out group keeps measuring what happens without action.</p>
  <p class="dg-caption">From predicted probability to a decision that minimises expected cost.</p>
</div>

## How it works: the math, step by step

### The cost matrix

Let $$y \in \{0, 1\}$$ be the true outcome (1 means the event happens, for example a failure) and let
$$\hat{y} \in \{0, 1\}$$ be our decision (1 means we act). There are four possible combinations, and
each has a cost:

| | Event happens ($$y=1$$) | Event does not happen ($$y=0$$) |
|---|---|---|
| We act ($$\hat{y}=1$$) | $$C_{TP}$$ | $$C_{FP}$$ |
| We do not act ($$\hat{y}=0$$) | $$C_{FN}$$ | $$C_{TN}$$ |

In plain English: TP is a correct alarm, FP is a false alarm, FN is a missed case, TN is correctly
doing nothing. Usually $$C_{TN} = 0$$.

### Expected cost of each choice

Suppose the model says the probability of the event is $$p = P(y = 1 \mid x)$$. If we act, the
expected cost is

$$
E[\text{cost} \mid \text{act}] = p \, C_{TP} + (1 - p) \, C_{FP}
$$

Read it as: with probability $$p$$ it was a real case and we pay the cost of a correct alarm; otherwise
it was a false alarm and we pay that cost instead.

If we do not act, the expected cost is

$$
E[\text{cost} \mid \text{wait}] = p \, C_{FN} + (1 - p) \, C_{TN}
$$

Read it as: with probability $$p$$ we miss a real case; otherwise nothing happens.

### The optimal threshold

We should act when acting is cheaper:

$$
p \, C_{TP} + (1 - p) \, C_{FP} < p \, C_{FN} + (1 - p) \, C_{TN}
$$

Move every term with $$p$$ to one side:

$$
(1 - p)(C_{FP} - C_{TN}) < p \, (C_{FN} - C_{TP})
$$

Solve for $$p$$:

$$
p > t^{*} = \frac{C_{FP} - C_{TN}}{(C_{FP} - C_{TN}) + (C_{FN} - C_{TP})}
$$

Line by line:

- The numerator, $$C_{FP} - C_{TN}$$, is how much extra a false alarm costs compared to correctly
  doing nothing. It is the **price of acting unnecessarily**.
- The second part of the denominator, $$C_{FN} - C_{TP}$$, is how much we save by catching a real case
  instead of missing it. It is the **value of acting correctly**.
- The threshold is the price of acting divided by the sum of price and value. When the value of
  catching a case is large compared to the price of a false alarm, the threshold drops towards zero
  and we act more often.

In the common special case where $$C_{TN} = 0$$ and $$C_{TP} = C_{FP} = c$$ (you pay the same action
cost whether the alarm was right or not) and a miss costs $$L$$, this becomes

$$
t^{*} = \frac{c}{L}
$$

In plain English: act whenever the probability of the event is higher than the ratio of action cost
to loss. If an inspection costs 1,000 and a failure costs 50,000, act whenever the probability is
above 2 percent.

### Why calibration matters

The formula assumes $$p$$ is a **true probability**: among all cases where the model says 2 percent,
roughly 2 percent should actually be events. Many models do not behave like that. Gradient boosting,
random forests and models trained with class weights or resampling often produce scores that rank
well but are not calibrated. We measure calibration with the Brier score

$$
\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

which is the average squared distance between the predicted probability and what actually happened.
Lower is better. If calibration is poor, we fit a small mapping from scores to probabilities (Platt
scaling or isotonic regression) on held-out data before applying the threshold.

### When costs differ per case

Sometimes the cost of a miss depends on the case. A failure on a large machine costs more than on a
small one. Then each case gets its own threshold:

$$
t_i^{*} = \frac{C_{FP,i} - C_{TN,i}}{(C_{FP,i} - C_{TN,i}) + (C_{FN,i} - C_{TP,i})}
$$

### When capacity is limited

Often the team can only act on $$K$$ cases per week. Then the question is not "which cases cross the
threshold" but "which $$K$$ cases give the largest expected saving". The expected saving of acting
on case $$i$$ is

$$
S_i = p_i \, (C_{FN,i} - C_{TP,i}) - (1 - p_i)(C_{FP,i} - C_{TN,i})
$$

which is the value of catching a real case, weighted by its probability, minus the price of a false
alarm, weighted by its probability. Rank by $$S_i$$, keep only cases with $$S_i > 0$$, and take the
top $$K$$.

## Implementation in Python

The module below keeps four concerns apart: the costs, calibration, threshold selection and the final
decision. Each piece can be tested on its own.

```python
# cost_decisions.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


@dataclass(frozen=True)
class CostMatrix:
    tp: float  # cost when we act and the event happens
    fp: float  # cost when we act and nothing happens
    fn: float  # cost when we do not act and the event happens
    tn: float = 0.0

    @property
    def analytic_threshold(self) -> float:
        price_of_acting = self.fp - self.tn
        value_of_acting = self.fn - self.tp
        if value_of_acting <= 0:
            raise ValueError("Acting never pays off: check the cost matrix")
        return price_of_acting / (price_of_acting + value_of_acting)


def expected_cost(y_true: np.ndarray, p: np.ndarray, threshold: float,
                  costs: CostMatrix) -> float:
    """Average realised cost per case for a given threshold."""
    act = p >= threshold
    tp = np.sum(act & (y_true == 1)) * costs.tp
    fp = np.sum(act & (y_true == 0)) * costs.fp
    fn = np.sum(~act & (y_true == 1)) * costs.fn
    tn = np.sum(~act & (y_true == 0)) * costs.tn
    return float((tp + fp + fn + tn) / len(y_true))


def empirical_threshold(y_true: np.ndarray, p: np.ndarray, costs: CostMatrix,
                        grid: np.ndarray | None = None) -> tuple[float, float]:
    """Search the threshold that minimises cost on a validation set."""
    grid = np.linspace(0.001, 0.999, 999) if grid is None else grid
    curve = np.array([expected_cost(y_true, p, t, costs) for t in grid])
    best = int(np.argmin(curve))
    return float(grid[best]), float(curve[best])


def calibrate(model, X_cal, y_cal, method: str = "isotonic"):
    """Wrap an already fitted model so its scores become probabilities."""
    from sklearn.frozen import FrozenEstimator  # scikit-learn >= 1.6

    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    return calibrated.fit(X_cal, y_cal)


def expected_saving(p: np.ndarray, costs: CostMatrix) -> np.ndarray:
    return p * (costs.fn - costs.tp) - (1 - p) * (costs.fp - costs.tn)


def select_top_k(p: np.ndarray, costs: CostMatrix, k: int) -> np.ndarray:
    """Indices of the k cases worth acting on when capacity is limited."""
    saving = expected_saving(p, costs)
    order = np.argsort(-saving)
    chosen = order[:k]
    return chosen[saving[chosen] > 0]


def calibration_report(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y_true, p)),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y_true.mean()),
    }
```

If you are on an older scikit-learn (before 1.6), replace the `FrozenEstimator` wrapper with
`CalibratedClassifierCV(model, method=method, cv="prefit")`.

A typical training script then reads like the business logic it implements:

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from cost_decisions import (CostMatrix, calibrate, calibration_report,
                            empirical_threshold, expected_cost)

# X, y: one row per machine-week, y = 1 if the machine failed in the following week
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=7)
X_cal, X_val, y_cal, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=7)

model = LGBMClassifier(n_estimators=400, learning_rate=0.05).fit(X_train, y_train)
model = calibrate(model, X_cal, y_cal)
p_val = model.predict_proba(X_val)[:, 1]

costs = CostMatrix(tp=2_000, fp=2_000, fn=60_000)
print(calibration_report(y_val, p_val))
print("analytic threshold ", round(costs.analytic_threshold, 4))
print("empirical threshold", empirical_threshold(y_val, p_val, costs))
print("cost at 0.5        ", expected_cost(y_val, p_val, 0.5, costs))
```

Two thresholds are printed on purpose. If the model is well calibrated, the analytic and empirical
thresholds should be close. If they are far apart, that is a signal to look at calibration before
going live.

## A worked example: predictive maintenance on a pump fleet

A water utility runs 1,200 pumps. Each week a model estimates the probability that each pump fails
within the next seven days. The costs, agreed with the operations team, are:

- An inspection (with a minor part replacement) costs 2,000 SEK, whether or not the pump was about
  to fail. So $$C_{TP} = C_{FP} = 2{,}000$$.
- An unplanned failure costs 60,000 SEK in emergency repair, overtime and service disruption. So
  $$C_{FN} = 60{,}000$$.
- Doing nothing to a healthy pump costs nothing. So $$C_{TN} = 0$$.

The threshold is

$$
t^{*} = \frac{2{,}000 - 0}{(2{,}000 - 0) + (60{,}000 - 2{,}000)} = \frac{2{,}000}{60{,}000} \approx 0.033
$$

So the right policy is to inspect every pump with a failure probability above about 3.3 percent, not
50 percent. With a typical weekly failure rate of around 1 percent, the default threshold of 0.5
would flag almost nothing and the utility would keep paying for emergency repairs.

Suppose the maintenance team can only do 40 inspections a week. We compute $$S_i$$ for each pump and
take the 40 with the largest positive saving. A pump with $$p = 0.20$$ has

$$
S = 0.20 \times 58{,}000 - 0.80 \times 2{,}000 = 11{,}600 - 1{,}600 = 10{,}000 \text{ SEK}
$$

of expected saving, so it goes to the top of the list. A pump with $$p = 0.02$$ has
$$S = 1{,}160 - 1{,}960 = -800$$, so inspecting it is not worth it even if there is spare capacity.

This gives the operations manager something they can actually use: a ranked list, an expected saving
per inspection, and a clear rule for when to stop.

## Pitfalls and when not to use it

- **Uncalibrated scores.** Applying a cost threshold to raw scores from a resampled or class-weighted
  model gives the wrong answer. Calibrate first, on data the model has not seen.
- **Costs nobody agreed on.** The threshold is only as good as the costs. Write them down with the
  business owner, and keep them in configuration rather than in code.
- **Costs that change.** Fuel prices, wages and contract penalties change. Recompute the threshold
  when they do, and log which cost version produced each decision.
- **Ignoring capacity.** A perfect threshold that flags 300 cases for a team that can handle 40 will
  be ignored in practice. Use the top-K rule.
- **Feedback loops.** If you always act on high-risk cases, you rarely observe what would have
  happened without action. Keep a small random hold-out group so you can keep measuring the model
  honestly.
- **When the decision is not binary.** If there are several possible actions (do nothing, remote
  check, site visit, replace), extend the cost matrix to one row per action and pick the action with
  the lowest expected cost.

## Measuring success in production

- Realised cost per period, compared with the cost under the previous policy.
- Precision and recall at the chosen threshold, reported monthly.
- Calibration: Brier score and a reliability plot per month.
- Share of flagged cases the team actually acted on (adoption).
- Drift in the predicted probability distribution, which often shows a problem before costs move.

## Checklist

- Agree the cost matrix with the business owner and store it in configuration.
- Split data into train, calibration and validation sets.
- Calibrate, then check the Brier score and a reliability plot.
- Compare the analytic and empirical thresholds.
- Add the capacity rule if the team cannot act on every flagged case.
- Keep a random hold-out group to measure the true effect over time.

## Final thought

Moving from "probability above 0.5" to "expected saving above zero" is usually a few dozen lines of
code, and in my experience it is often worth more than another month of model tuning. It also changes
the conversation with stakeholders: instead of debating accuracy, you discuss costs they understand.
If you want help putting a decision layer like this on top of an existing model, see my
[real-time ML and decision systems service](/services/#real-time-ml-systems) or
[contact me](/contact/).
