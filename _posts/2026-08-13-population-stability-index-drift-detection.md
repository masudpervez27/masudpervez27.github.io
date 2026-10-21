---
title: "Population Stability Index in Practice: Detecting Data Drift Before It Hurts Your Model"
date: 2026-08-13T10:00:00+02:00
excerpt: "PSI is the most widely used drift metric in production ML, and also one of the most misread. Where it comes from, why the famous 0.1 and 0.25 limits break on small and large samples, how to set limits that depend on sample size, and a modular Python drift monitor."
categories:
  - MLOps
  - Machine Learning
tags:
  - Data Drift
  - Population Stability Index
  - Model Monitoring
  - Statistics
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "chart-area"
---

Every model I have put into production has met data it was not trained on. Customers change,
prices change, an upstream system starts sending a new category code, a marketing campaign brings in
a different kind of user. The model keeps scoring without complaint, and the first sign of trouble is
often a business result that looks wrong weeks later. The Population Stability Index (PSI) is the
simplest reliable way I know to see that change coming. In this post I explain where PSI comes from,
why the usual rules of thumb mislead on small and very large samples, and how I build a drift monitor
that the business can trust.

## The business problem

A model is a summary of the past. It works as long as the future looks enough like the data it
learned from. When the input data shifts, three things can happen:

- **Silent degradation.** Scores move and decisions get worse, but nothing errors. A churn model
  starts flagging the wrong customers, or a demand forecast drifts high after a change in the
  product range.
- **Broken data that looks valid.** A unit changes from kronor to öre, a category is renamed, a field
  starts arriving empty. The pipeline runs, the model scores, and the results are nonsense.
- **Late discovery.** True outcomes (did the customer leave, what was actual demand) often arrive
  weeks or months later. Waiting for them means the business has been acting on bad scores for that
  whole period.

Drift monitoring on the *inputs* gives an early warning that does not depend on waiting for labels.
It also gives the team a concrete, explainable reason to retrain, rather than retraining on a fixed
calendar whether it is needed or not.

## The idea in plain English

Take a feature, for example "monthly spend". From a reference period (usually the training data or a
stable recent window) cut its values into ten groups of equal size, so each group holds 10 percent of
the reference rows. Then take this week's data and count what share falls into each of the same
groups.

If nothing has changed, each group should still hold about 10 percent. If customers are suddenly
spending more, the top groups fill up and the bottom groups empty out. PSI turns those differences
into one number per feature: zero when the shares match exactly, and larger the more they have moved.

The second half of the idea is just as important. Even when nothing has truly changed, random
sampling makes the shares wobble a little, so PSI is never exactly zero. With small samples the
wobble is large; with very large samples even meaningless changes look "significant". A good monitor
sets its alert levels with that in mind.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Reference window<small>training or stable period</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Bins from reference<small>deciles or categories</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Current data<small>counted into the same bins</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">PSI per feature</div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Limits<small>sample-size aware</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-branch">
      <div class="dg-node dg-ok">Stable: log only</div>
      <div class="dg-node dg-warn">Shift: alert and investigate</div>
    </div>
  </div>
  <p class="dg-caption">A weekly drift check on every important feature and on the model score.</p>
</div>

## How it works: the math, step by step

### The definition

Split the feature into $$B$$ bins using the reference data. Let $$r_b$$ be the share of reference
rows in bin $$b$$ and $$a_b$$ the share of current ("actual") rows in the same bin. Then

$$
\text{PSI} = \sum_{b=1}^{B} (a_b - r_b) \, \ln\!\left(\frac{a_b}{r_b}\right)
$$

Line by line:

- $$a_b - r_b$$ is how much the share in bin $$b$$ moved, in percentage points.
- $$\ln(a_b / r_b)$$ is how big that move is *relative to* the original share. Moving from 1 to 3
  percent is a much bigger change than moving from 30 to 32 percent, and the log captures that.
- Summing over all bins gives one number for the feature.

### Why every term is positive

If $$a_b > r_b$$, both factors are positive. If $$a_b < r_b$$, both are negative. Either way their
product is positive, and it is zero only when $$a_b = r_b$$. So PSI is never negative, and it is zero
only when the two distributions agree on every bin.

### Where PSI comes from

The Kullback-Leibler divergence measures how different one distribution is from another:

$$
\text{KL}(a \,\|\, r) = \sum_{b} a_b \ln\frac{a_b}{r_b}
\qquad
\text{KL}(r \,\|\, a) = \sum_{b} r_b \ln\frac{r_b}{a_b} = -\sum_{b} r_b \ln\frac{a_b}{r_b}
$$

KL is not symmetric: comparing $$a$$ to $$r$$ gives a different number from comparing $$r$$ to
$$a$$. Adding the two directions gives

$$
\text{KL}(a \,\|\, r) + \text{KL}(r \,\|\, a) = \sum_{b} (a_b - r_b) \ln\frac{a_b}{r_b} = \text{PSI}
$$

In plain English: PSI is the symmetric version of KL divergence, sometimes called the Jeffreys
divergence. It treats "the data moved away from the reference" and "the reference moved away from the
data" the same way, which is exactly what you want from a monitoring metric.

### How large PSI is when nothing has changed

Suppose the reference has $$n$$ rows, the current window has $$m$$ rows, and the true distribution
has not changed at all. For small differences, $$\ln(a_b / r_b) \approx (a_b - r_b) / r_b$$, so

$$
\text{PSI} \approx \sum_{b=1}^{B} \frac{(a_b - r_b)^2}{r_b}
$$

This is almost the Pearson chi-square statistic. The sampling variance of $$a_b - r_b$$ is about
$$r_b (1 - r_b)\left(\tfrac{1}{n} + \tfrac{1}{m}\right)$$, which leads to the approximation

$$
\text{PSI} \approx \left(\frac{1}{n} + \frac{1}{m}\right) \chi^2_{B-1}
$$

Read it as: under no real change, PSI behaves like a chi-square random variable with $$B - 1$$
degrees of freedom, scaled down by the sample sizes. Two practical consequences follow.

The **expected PSI from noise alone** is

$$
E[\text{PSI}] \approx (B - 1)\left(\frac{1}{n} + \frac{1}{m}\right)
$$

With ten bins and 200 rows on each side, that is $$9 \times 0.01 = 0.09$$. In other words, with small
windows you will see PSI values near 0.1 *every week* even when nothing has changed.

A **statistical alert limit** at confidence level $$1 - \alpha$$ is

$$
\text{PSI}_{\text{crit}} = \left(\frac{1}{n} + \frac{1}{m}\right) \chi^2_{B-1,\,1-\alpha}
$$

With $$B = 10$$ and $$\alpha = 0.01$$, the chi-square quantile is about 21.67. For 200 rows on each
side, the limit is about 0.22. For a million rows on each side, it is about 0.00004.

### Statistical versus practical significance

That last number shows the opposite problem. With very large samples, tiny and harmless shifts become
statistically significant. The fixed industry limits (below 0.1 stable, 0.1 to 0.25 moderate, above
0.25 significant) are really *practical* thresholds: they say how large a shift usually matters for a
model. I combine both:

$$
\text{limit} = \max\left(\text{PSI}_{\text{crit}},\ \text{PSI}_{\text{practical}}\right)
$$

In plain English: only alert when a shift is both larger than random noise *and* large enough to
matter. On small windows the statistical limit protects you from false alarms; on large windows the
practical limit stops you chasing trivial changes.

### Bins, empty bins and categories

- **Numeric features:** use quantiles of the reference (deciles are standard). Every bin then starts
  with the same share, and no bin is nearly empty. Set the outer edges to minus and plus infinity so
  new extreme values still land in a bin.
- **Empty bins:** if $$a_b = 0$$ the log is undefined. Add a small $$\varepsilon$$ (for example
  0.0001) to every share before computing.
- **Categorical features:** each category is a bin. Group rare categories into "other", and give
  missing values and unseen categories their own bins. A brand new category appearing is itself a
  drift signal.

### What PSI cannot see

PSI compares *input distributions*. It cannot see a change in the relationship between inputs and
outcome (concept drift), for example the same customers behaving differently after a competitor
launches. That still needs monitoring of model performance once true outcomes arrive. PSI is the early
warning; performance monitoring is the confirmation.

## Implementation in Python

### Configuration and core functions

```python
# drift/psi.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DriftConfig:
    bins: int = 10
    eps: float = 1e-4
    alpha: float = 0.01
    practical_warn: float = 0.10
    practical_alert: float = 0.25
    max_categories: int = 20


def quantile_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(reference[~np.isnan(reference)], np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def numeric_shares(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Shares per bin, with missing values as an extra last bin."""
    missing = np.isnan(values)
    counts = np.histogram(values[~missing], bins=edges)[0]
    counts = np.append(counts, missing.sum())
    return counts / max(len(values), 1)


def categorical_shares(values: pd.Series, categories: list[str]) -> np.ndarray:
    """Shares per known category, plus 'other' and 'missing'."""
    s = values.astype("object")
    known = s.isin(categories)
    counts = [int((s == c).sum()) for c in categories]
    counts.append(int((~known & s.notna()).sum()))
    counts.append(int(s.isna().sum()))
    return np.array(counts) / max(len(s), 1)


def psi(ref_shares: np.ndarray, cur_shares: np.ndarray, eps: float = 1e-4) -> float:
    r = ref_shares + eps
    a = cur_shares + eps
    r, a = r / r.sum(), a / a.sum()
    return float(np.sum((a - r) * np.log(a / r)))


def psi_critical(n_ref: int, n_cur: int, n_bins: int, alpha: float) -> float:
    """Upper limit of PSI expected from sampling noise alone."""
    return (1 / n_ref + 1 / n_cur) * stats.chi2.ppf(1 - alpha, df=n_bins - 1)
```

### The monitor

```python
# drift/monitor.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from drift.psi import (DriftConfig, categorical_shares, numeric_shares, psi,
                       psi_critical, quantile_edges)


@dataclass
class FeatureProfile:
    kind: str                      # "numeric" or "categorical"
    ref_shares: np.ndarray
    n_ref: int
    edges: np.ndarray | None = None
    categories: list[str] = field(default_factory=list)


class DriftMonitor:
    def __init__(self, cfg: DriftConfig = DriftConfig()):
        self.cfg = cfg
        self.profiles: dict[str, FeatureProfile] = {}

    def fit(self, reference: pd.DataFrame) -> "DriftMonitor":
        for col in reference.columns:
            s = reference[col]
            if pd.api.types.is_numeric_dtype(s):
                values = s.to_numpy(dtype=float)
                edges = quantile_edges(values, self.cfg.bins)
                shares = numeric_shares(values, edges)
                self.profiles[col] = FeatureProfile("numeric", shares, len(s), edges=edges)
            else:
                top = s.value_counts().head(self.cfg.max_categories).index.astype(str).tolist()
                shares = categorical_shares(s.astype("object").where(s.isna(), s.astype(str)), top)
                self.profiles[col] = FeatureProfile("categorical", shares, len(s), categories=top)
        return self

    def _current_shares(self, profile: FeatureProfile, s: pd.Series) -> np.ndarray:
        if profile.kind == "numeric":
            return numeric_shares(s.to_numpy(dtype=float), profile.edges)
        return categorical_shares(s.astype("object").where(s.isna(), s.astype(str)), profile.categories)

    def check(self, current: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col, profile in self.profiles.items():
            cur = self._current_shares(profile, current[col])
            value = psi(profile.ref_shares, cur, self.cfg.eps)
            n_bins = int(np.count_nonzero(profile.ref_shares + cur))
            crit = psi_critical(profile.n_ref, len(current), max(n_bins, 2), self.cfg.alpha)
            warn = max(crit, self.cfg.practical_warn)
            alert = max(crit, self.cfg.practical_alert)
            status = "alert" if value > alert else "warn" if value > warn else "stable"
            rows.append({"feature": col, "psi": round(value, 4), "noise_limit": round(crit, 4),
                         "status": status, "n_current": len(current)})
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def top_movers(profile: FeatureProfile, current: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Which bins contributed most to PSI, to explain an alert."""
    r, a = profile.ref_shares + 1e-4, current + 1e-4
    contrib = (a - r) * np.log(a / r)
    order = np.argsort(-contrib)[:k]
    return [(int(i), float(contrib[i])) for i in order]
```

### Weekly use

```python
import pandas as pd

from drift.monitor import DriftMonitor

features = ["monthly_price", "sessions_30d", "support_tickets_90d", "plan", "country"]
reference = training_df[features]
monitor = DriftMonitor().fit(reference)

this_week = scoring_df.loc[scoring_df["week"] == "2026-W32", features]
report = monitor.check(this_week)
print(report)

alerts = report[report["status"] == "alert"]
if not alerts.empty:
    print("Investigate:", ", ".join(alerts["feature"]))
```

I always monitor the **model score** itself in the same way. If the score distribution moves while
no single feature crosses a limit, several small shifts are adding up, and that is worth knowing.

## A worked example: a subscription service after a price change

A subscription business uses a churn model trained on last year's data. In August it raises prices
for new customers. The weekly drift check uses a reference of 50,000 rows and a current window of
12,000 rows.

To keep the arithmetic readable, look at the feature `monthly_price` with four bins (quartiles of the
reference), so each reference share is 0.25. This week's shares are 0.10, 0.20, 0.30 and 0.40:

$$
\begin{aligned}
\text{PSI} &= (0.10 - 0.25)\ln\tfrac{0.10}{0.25} + (0.20 - 0.25)\ln\tfrac{0.20}{0.25}
            + (0.30 - 0.25)\ln\tfrac{0.30}{0.25} + (0.40 - 0.25)\ln\tfrac{0.40}{0.25} \\
           &\approx 0.137 + 0.011 + 0.009 + 0.071 = 0.228
\end{aligned}
$$

Line by line: the lowest price band lost most of its share (contributing 0.137), and the highest band
gained the most (0.071). The middle bands barely moved.

The noise limit with three degrees of freedom at 99 percent confidence is

$$
\left(\frac{1}{50{,}000} + \frac{1}{12{,}000}\right) \times 11.34 \approx 0.0012
$$

so the shift is far beyond random noise. Against the practical limits it is a "warn", close to
"alert". The `top_movers` output points straight at the low price band, which matches the known
price change.

The decision is not automatic retraining. The team checks whether the model's score distribution has
also moved (it has, slightly upwards), and whether the price change makes last year's churn patterns
less relevant for new customers (it does). They schedule a retrain once four weeks of post-change
outcomes are available, and in the meantime apply a temporary recalibration. That is a calm,
evidence-based response instead of a surprise in the quarterly churn report.

## Pitfalls and when not to use it

- **Fixed limits on small windows.** With a few hundred rows, PSI around 0.1 is normal noise. Use
  the sample-size-aware limit, or use larger windows.
- **Seasonality mistaken for drift.** December always looks different from June. Compare against the
  same period last year, or use a rolling reference that includes the season.
- **Too many bins.** More bins means more noise per bin. Ten is a good default; five for small
  samples.
- **Changing the bins over time.** Bins must come from the reference and stay fixed. Recomputing them
  on current data hides the drift you are trying to see.
- **Alert fatigue.** Monitoring 300 features at 99 percent confidence still produces about three
  false alerts per run. Focus on the features the model actually relies on, and alert on the score.
- **Treating PSI as a performance metric.** A stable PSI does not mean the model is still accurate.
  Concept drift needs outcome-based monitoring.

## Measuring success in production

- Time from a real upstream change to its detection.
- Share of alerts that led to a confirmed issue (alert precision).
- Number of incidents found first by the business instead of by monitoring (target: zero).
- Model performance before and after drift-triggered retraining.

## Checklist

- Choose a stable reference window and store its bins with the model version.
- Use quantile bins with open outer edges, plus a missing-value bin.
- Treat categories, "other" and missing values as bins for categorical features.
- Compute the noise limit from sample sizes and combine it with practical limits.
- Monitor the model score as well as the inputs.
- Explain each alert with the bins that moved most.
- Pair PSI with performance monitoring once true outcomes arrive.

## Final thought

PSI is simple, and that is its strength. Used with fixed bins, sample-size-aware limits and a clear
explanation of which bins moved, it gives the business an early, honest signal that the world has
changed. It pairs naturally with the pipeline design in my post on
[reliable feature pipelines](/posts/2026-02-20-scaling-production-mlops-spark-feature-pipelines/).
If you want monitoring like this around your own models, see my
[production MLOps service](/services/#production-mlops) or [get in touch](/contact/).
