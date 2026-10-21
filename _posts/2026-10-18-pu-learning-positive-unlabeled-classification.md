---
title: "PU Learning: Building a Classifier When You Only Know the Positives"
date: 2026-10-18T10:00:00+02:00
excerpt: "Many real datasets have confirmed positives and a large pile of unknowns, not clean positive and negative labels. Positive-unlabelled (PU) learning turns that situation into calibrated probabilities. The assumption that makes it work, the correction formula derived step by step, a reusable Python wrapper, and a dangerous-goods classification example."
categories:
  - Machine Learning
  - Statistics
tags:
  - PU Learning
  - Semi-Supervised Learning
  - Classification
  - Probability Calibration
  - scikit-learn
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "circle-half-stroke"
---

Textbook classification assumes you have examples of both classes: these are positives, those are
negatives. In practice I often get something else. There is a list of confirmed cases, and then
everything else, which nobody has checked. The "everything else" is not negative. It is unknown, and
some of it is positive. If you train a normal classifier on that data, it learns to separate
*labelled* from *unlabelled*, which is not the question you care about. Positive-unlabelled (PU)
learning fixes this with a surprisingly small correction. In this post I explain when it works, derive
the correction step by step, and show a Python wrapper you can put around any scikit-learn model.

## The business problem

PU data shows up wherever confirming a positive is easy, but confirming a negative is expensive or
never happens:

- **Product data quality.** Some products in a catalogue are flagged as dangerous goods, age
  restricted or fragile. Many others should be flagged but were not.
- **Equipment faults.** Confirmed faults are logged by technicians. Machines without a log entry are
  not necessarily healthy; their faults may simply not have been found yet.
- **Customer interest.** Customers who bought an add-on are known positives. Customers who did not
  buy were often never offered it.
- **Document classification.** A team tags documents that contain a certain clause. Untagged
  documents were mostly never reviewed.

Treating the unknowns as negatives has a real cost. The model learns that the hidden positives look
"negative", it underestimates risk or interest everywhere, and the business acts on probabilities that
are systematically too low. Worse, nobody can tell, because the evaluation uses the same wrong labels.

## The idea in plain English

Imagine that a fixed share of all true positives gets labelled, say 30 percent, and that which ones
get labelled is essentially random. Then:

1. Train an ordinary classifier to predict "is this example labelled?". Call its output the labelling
   score.
2. For a true positive, the chance of being labelled is about 30 percent. So the labelling score of a
   typical true positive will be about 0.30, never close to 1.
3. Measure that number on known positives the model did not train on. That gives the labelling rate.
4. Divide every labelling score by the labelling rate. A product with a labelling score of 0.12 has a
   true probability of being positive of about $$0.12 / 0.30 = 0.40$$.

That is the whole trick: a classifier for "labelled or not", plus one number that rescales it into "positive or
not". The rest of this post is about why it works, when it does not, and how to evaluate a model
without any confirmed negatives.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-branch">
      <div class="dg-node">Labelled positives</div>
      <div class="dg-node">Unlabelled pool<small>mix of positives and negatives</small></div>
    </div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Classifier<small>labelled vs unlabelled</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">Estimate labelling rate<small>on held-out positives</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Corrected probability<small>score divided by rate</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Rank and choose<small>within review capacity</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-warn">Spot-check sample<small>estimates precision</small></div>
  </div>
  <p class="dg-caption">Turning "labelled or not" into "positive or not" with one estimated rate.</p>
</div>

## How it works: the math, step by step

### Notation

For each example $$x$$:

- $$y \in \{0, 1\}$$ is the true class (1 means positive), which we do not observe for unlabelled
  examples.
- $$s \in \{0, 1\}$$ says whether the example is labelled. Only positives can be labelled, so
  $$P(s = 1 \mid y = 0) = 0$$.

### The key assumption: labelled completely at random (SCAR)

$$
P(s = 1 \mid x, \, y = 1) = P(s = 1 \mid y = 1) = c
$$

In plain English: among the true positives, whether an example got labelled does not depend on its
features. Every true positive has the same chance $$c$$ of being labelled. This is the assumption
that makes the problem solvable, and also the one to question first in any real project.

### Deriving the correction

Because only positives can be labelled,

$$
P(s = 1 \mid x) = P(s = 1 \mid x, y = 1)\, P(y = 1 \mid x) + P(s = 1 \mid x, y = 0)\, P(y = 0 \mid x)
$$

The second term is zero, and SCAR replaces the first factor with $$c$$:

$$
P(s = 1 \mid x) = c \cdot P(y = 1 \mid x)
$$

Rearranging gives the result we want:

$$
P(y = 1 \mid x) = \frac{P(s = 1 \mid x)}{c} = \frac{g(x)}{c}
$$

where $$g(x)$$ is the classifier trained to predict $$s$$. Line by line:

- $$g(x)$$ is something we can learn directly, because $$s$$ is fully observed.
- Dividing by $$c$$ scales it up from "probability of being labelled" to "probability of being
  positive".
- The ranking of examples does not change, only the scale. So if you only need a ranking, the
  labelled-versus-unlabelled classifier is already useful. You need $$c$$ to get *calibrated*
  probabilities and to estimate how many positives exist in total.

### Estimating the labelling rate

Take a validation set $$P_{\text{val}}$$ of labelled positives that $$g$$ was not trained on. For a
true positive whose features clearly mark it as positive, $$P(y = 1 \mid x) \approx 1$$, so
$$g(x) \approx c$$. Averaging over the held-out positives gives the estimator

$$
\hat{c} = \frac{1}{\lvert P_{\text{val}} \rvert} \sum_{x \in P_{\text{val}}} g(x)
$$

In plain English: look at how confident the labelled-versus-unlabelled model is on positives it has
never seen. That average confidence is the share of positives that get labelled. This is the
estimator from Elkan and Noto (2008). It needs $$g$$ to be a calibrated probability, so I always
calibrate the base model.

### How many positives are there?

The share of labelled examples overall is $$P(s = 1) = c \cdot P(y = 1)$$, so the class prior is

$$
\pi = P(y = 1) = \frac{P(s = 1)}{c}
$$

In plain English: if 3 percent of all examples are labelled and only 30 percent of positives get
labelled, then about 10 percent of all examples are truly positive. This one number is often the most
valuable output for the business, because it sizes the hidden problem.

### Weighting the unlabelled examples

An alternative to rescaling is to train a final classifier where each unlabelled example appears
twice: once as a positive with weight $$w(x)$$ and once as a negative with weight $$1 - w(x)$$, where

$$
w(x) = P(y = 1 \mid x, s = 0) = \frac{1 - c}{c} \cdot \frac{g(x)}{1 - g(x)}
$$

Read it as: among examples that were *not* labelled, the chance of being a hidden positive grows with
the labelling score, and is larger when the labelling rate $$c$$ is small (many positives were
missed). This version tends to give a better final model, because the second classifier learns from
cleaner soft labels.

### Learning directly from PU data: the unbiased risk

A third approach changes the loss function instead. For a classifier $$f$$ and loss $$\ell$$, the
true risk can be written using only positive and unlabelled data:

$$
R(f) = \pi\, E_{P}\big[\ell(f(x), +1)\big] + \Big( E_{U}\big[\ell(f(x), -1)\big] - \pi\, E_{P}\big[\ell(f(x), -1)\big] \Big)
$$

Line by line:

- The first term is the loss on positives, weighted by how common positives are.
- $$E_{U}[\ell(f(x), -1)]$$ treats all unlabelled data as negative, which counts the hidden positives
  wrongly as negatives.
- Subtracting $$\pi\, E_{P}[\ell(f(x), -1)]$$ removes exactly that wrong part.

With flexible models the bracket can go below zero during training, which means overfitting. The
**non-negative PU** estimator (nnPU, Kiryo and colleagues, 2017) clips it at zero:

$$
R_{\text{nn}}(f) = \pi\, E_{P}\big[\ell(f(x), +1)\big] + \max\Big(0,\; E_{U}\big[\ell(f(x), -1)\big] - \pi\, E_{P}\big[\ell(f(x), -1)\big]\Big)
$$

In plain English: the estimated loss on negatives can never be negative, so the model is stopped
from memorising the positives. This is the approach I use for neural networks; for tabular data the
simpler two-step method above is usually enough.

### Evaluating without negatives

We cannot compute precision directly, because we do not know which unlabelled examples are negative.
But recall on held-out labelled positives is still measurable, because under SCAR they are a random
sample of all positives. Lee and Liu (2003) noticed that

$$
\frac{r^2}{P(\hat{y} = 1)} = \frac{p \cdot r}{\pi}
$$

where $$r$$ is recall, $$p$$ is precision and $$P(\hat{y} = 1)$$ is the share of examples predicted
positive. The derivation is short: precision is $$p = P(y = 1 \mid \hat{y} = 1) = r \pi / P(\hat{y} = 1)$$,
so $$p \cdot r / \pi = r^2 / P(\hat{y} = 1)$$. Since $$\pi$$ is fixed, maximising this score is the
same as maximising $$p \cdot r$$, and every part of it can be computed from PU data. I use it to
compare models and thresholds, then confirm the final choice with a small manual spot check.

## Implementation in Python

### A PU wrapper around any classifier

```python
# pu/classifier.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


@dataclass
class PUConfig:
    holdout_fraction: float = 0.2
    calibration_cv: int = 3
    random_state: int = 7


class PUClassifier(BaseEstimator, ClassifierMixin):
    """Elkan-Noto PU learning: learn P(labelled | x), then rescale by the labelling rate c."""

    def __init__(self, base_estimator=None, cfg: PUConfig | None = None):
        self.base_estimator = base_estimator
        self.cfg = cfg

    def fit(self, X: np.ndarray, s: np.ndarray) -> "PUClassifier":
        cfg = self.cfg or PUConfig()
        s = np.asarray(s).astype(int)
        base = self.base_estimator or HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05)

        pos_idx = np.flatnonzero(s == 1)
        train_pos, holdout_pos = train_test_split(
            pos_idx, test_size=cfg.holdout_fraction, random_state=cfg.random_state
        )
        train_idx = np.concatenate([train_pos, np.flatnonzero(s == 0)])

        self.model_ = CalibratedClassifierCV(clone(base), method="isotonic", cv=cfg.calibration_cv)
        self.model_.fit(X[train_idx], s[train_idx])

        g_holdout = self.model_.predict_proba(X[holdout_pos])[:, 1]
        self.c_ = float(np.clip(g_holdout.mean(), 1e-6, 1.0))
        self.prior_ = float(np.clip(s.mean() / self.c_, 0.0, 1.0))
        self.classes_ = np.array([0, 1])
        return self

    def labelling_score(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.labelling_score(X) / self.c_, 0.0, 1.0)
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def hidden_positive_weight(self, X: np.ndarray) -> np.ndarray:
        """P(y=1 | x, s=0): the soft label for an unlabelled example."""
        g = np.clip(self.labelling_score(X), 1e-6, 1 - 1e-6)
        return np.clip((1 - self.c_) / self.c_ * g / (1 - g), 0.0, 1.0)
```

### Evaluation helpers

```python
# pu/evaluation.py
from __future__ import annotations

import numpy as np


def recall_on_positives(p_pos: np.ndarray, threshold: float) -> float:
    """Recall measured on held-out labelled positives (valid under SCAR)."""
    return float(np.mean(p_pos >= threshold))


def lee_liu_score(p_pos: np.ndarray, p_all: np.ndarray, threshold: float) -> float:
    """r^2 / P(predicted positive): proportional to precision x recall."""
    r = recall_on_positives(p_pos, threshold)
    share_pred = float(np.mean(p_all >= threshold))
    return r * r / share_pred if share_pred > 0 else 0.0


def best_threshold(p_pos: np.ndarray, p_all: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.05, 0.95, 91)
    scores = [lee_liu_score(p_pos, p_all, t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def spot_check_precision(confirmed_positive: int, checked: int, z: float = 1.96) -> tuple[float, float, float]:
    """Precision with a Wilson interval from a manual review sample."""
    p = confirmed_positive / checked
    denom = 1 + z**2 / checked
    centre = (p + z**2 / (2 * checked)) / denom
    half = z * np.sqrt(p * (1 - p) / checked + z**2 / (4 * checked**2)) / denom
    return p, centre - half, centre + half
```

### Putting it together

```python
import numpy as np

from pu.classifier import PUClassifier
from pu.evaluation import best_threshold, spot_check_precision

# X: product features; s: 1 if a shipper flagged the item as dangerous goods, else 0
pu = PUClassifier().fit(X, s)
print(f"labelling rate c = {pu.c_:.2f}, estimated share of true positives = {pu.prior_:.3f}")

p_all = pu.predict_proba(X)[:, 1]
unlabelled = np.flatnonzero(s == 0)
review_queue = unlabelled[np.argsort(-p_all[unlabelled])][:500]

# after the compliance team checks 200 of the top-ranked items
print(spot_check_precision(confirmed_positive=150, checked=200))
```

In a real project I also keep the held-out positives from `fit` to compute recall and the Lee and Liu
score with `best_threshold`, so that model comparisons use the same data.

## A worked example: finding unflagged dangerous goods

A logistics company handles 200,000 product lines (SKUs) from many shippers. Shippers are supposed to
flag dangerous goods (batteries, aerosols, flammable liquids), because those need special handling
and documentation. 6,000 SKUs are flagged. Audits suggest that many others should be.

- The share of labelled items is $$P(s = 1) = 6{,}000 / 200{,}000 = 0.03$$.
- The labelled-versus-unlabelled model, scored on 1,200 held-out flagged SKUs, gives an average
  labelling score of $$\hat{c} = 0.30$$. Only about 30 percent of true dangerous goods get flagged.
- The estimated share of true dangerous goods is
  $$\pi = 0.03 / 0.30 = 0.10$$, about 20,000 SKUs. So roughly 14,000 dangerous items are moving
  through the network without the right handling.

That number alone changes the conversation with management. It is no longer "some shippers forget to
flag things" but "about 14,000 product lines are probably handled incorrectly".

For an individual unflagged SKU with a labelling score of 0.12, the corrected probability is
$$0.12 / 0.30 = 0.40$$, and its hidden-positive weight is

$$
w = \frac{1 - 0.30}{0.30} \cdot \frac{0.12}{1 - 0.12} = 2.33 \times 0.136 \approx 0.32
$$

The compliance team can review 500 SKUs a month, so the model ranks all unflagged SKUs and sends the
top 500. In the first batch, they check 200 and confirm 150 as dangerous goods: a precision of 75
percent, with a 95 percent interval of roughly 69 to 80 percent. Without the model, a random review
would find dangerous goods in about 7 percent of unflagged items (14,000 out of 194,000). The ranked
list makes each hour of review about ten times more productive, and every confirmed item becomes a
new labelled positive for the next training round.

## Pitfalls and when not to use it

- **Labelling that is not random.** If shippers flag obvious items (big lithium batteries) but miss
  subtle ones (small cells inside devices), SCAR fails and $$\hat{c}$$ is too high, so probabilities
  are too low for the subtle cases. Check whether labelled positives look systematically different
  from confirmed hidden positives, and include features that explain *who* labels.
- **Too few labelled positives.** With a few dozen positives, the estimate of $$c$$ is noisy. Use a
  larger hold-out or cross-validation and report an interval.
- **Uncalibrated base models.** The estimator of $$c$$ assumes $$g$$ is a calibrated probability.
  Always calibrate.
- **Corrected probabilities above 1.** They appear when $$c$$ is underestimated or $$g$$ is badly
  calibrated in the tail. Clipping hides the symptom; investigate the cause.
- **When negatives are cheap to get.** If you can label a random sample of 1,000 unlabelled examples
  in a week, do that. Real negatives beat clever assumptions.
- **When positives can come from several processes.** If labels come from very different sources
  (an automatic rule and manual reports), estimate $$c$$ per source or model them separately.

## Measuring success in production

- Precision of the review queue, measured by regular spot checks.
- Number of hidden positives found per review hour, compared with random sampling.
- Stability of $$\hat{c}$$ and $$\pi$$ over retraining rounds.
- Business outcome: incidents, fines or rework linked to unflagged positives.

## Checklist

- Confirm that unlabelled really means "unknown", not "checked and negative".
- Question the SCAR assumption with the people who create the labels.
- Train a calibrated classifier for labelled versus unlabelled.
- Estimate the labelling rate on held-out positives and report the implied class prior.
- Rank unlabelled examples and fit the review queue to the team's capacity.
- Evaluate with recall on positives, the Lee and Liu score, and manual spot checks.
- Feed confirmed positives back into training.

## Final thought

PU learning is one of those methods that looks academic until you notice how many business datasets
are secretly positive-unlabelled. Recognising that, and correcting for it with one well-estimated
number, can change both the model and the size of the problem the business thinks it has. If you
recognise this pattern in your own data, see my
[forecasting, optimisation and decision analytics service](/services/#forecasting-decision-analytics) or
[get in touch](/contact/).
