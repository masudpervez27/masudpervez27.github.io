---
title: "Probabilistic Demand Forecasting for Logistics: Quantile Models, Pinball Loss and Calibrated Intervals"
date: 2026-05-14T10:00:00+02:00
excerpt: "A single-number forecast hides the information planners need most: how wrong it could be. How to forecast ranges instead of points with quantile gradient boosting, how to check those ranges are honest with conformal calibration, and how a parcel hub turns them into staffing decisions."
categories:
  - Forecasting
  - Logistics
tags:
  - Probabilistic Forecasting
  - Quantile Regression
  - LightGBM
  - Conformal Prediction
  - Supply Chain
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "truck"
---

Every logistics planner I have worked with already knows that the forecast will be wrong. What they
actually need to know is *how* wrong, and in which direction it is safer to be wrong. A forecast that
says "12,000 parcels tomorrow" does not help you decide whether to book three extra drivers. A
forecast that says "80 percent chance between 10,900 and 13,600, and 1 in 10 chance above 14,200"
does. This post explains how to build that second kind of forecast, how to check that its ranges can
be trusted, and how to turn it into decisions.

## The business problem

Logistics operations commit resources before demand arrives:

- Warehouse and sorting staff are scheduled days in advance.
- Trucks, drivers and carrier capacity are booked ahead.
- Stock is positioned in regional hubs before orders come in.

Each decision has asymmetric costs. Too little capacity means overtime, late deliveries and missed
service levels. Too much means paying people and vehicles to wait. A point forecast cannot balance
those costs because it does not say how likely each outcome is. Planners compensate with private
buffers ("I always add 15 percent on Mondays"), which are inconsistent, invisible to management and
usually too large on calm days and too small on busy ones.

A probabilistic forecast makes uncertainty explicit, so the buffer can be chosen from the costs
rather than from habit.

## The idea in plain English

Instead of predicting one number, we predict several **quantiles**. The 10th percentile (P10) is a
level that demand should fall below only 10 percent of the time. The 50th (P50) is the median. The
90th (P90) is a level demand should exceed only 10 percent of the time.

Together they describe a range and its shape. Quiet weeks get narrow ranges; the week before a
holiday gets a wide one. A planner can then choose the quantile that matches the cost of being short
versus the cost of being over, using exactly the logic of the
[newsvendor model](/posts/2026-04-05-maximizing-profit-margins-stochastic-newsvendor-optimization/).

The second half of the idea is honesty. If the P90 is exceeded 25 percent of the time instead of 10,
the ranges are lying. We fix that with a calibration step that adjusts the ranges using recent
errors, so a "90 percent" range really covers about 90 percent of outcomes.

## How it works: the math, step by step

### What a quantile is

For a random variable $$Y$$ with distribution $$F$$, the quantile at level $$\tau \in (0, 1)$$ is

$$
q_{\tau} = F^{-1}(\tau) = \inf\{\, y : F(y) \ge \tau \,\}
$$

In plain English: the smallest value such that at least a share $$\tau$$ of outcomes are at or below
it. A conditional quantile $$q_{\tau}(x)$$ does the same thing given features $$x$$ such as weekday,
season and recent volumes.

### The pinball loss

To train a model that predicts the mean, we minimise squared error. To train a model that predicts a
quantile, we minimise the **pinball loss** (also called quantile loss):

$$
L_{\tau}(y, \hat{q}) =
\begin{cases}
\tau \,(y - \hat{q}) & \text{if } y \ge \hat{q} \\
(1 - \tau)\,(\hat{q} - y) & \text{if } y < \hat{q}
\end{cases}
$$

Line by line:

- If the actual value is above the prediction (we under-forecast), the error is multiplied by
  $$\tau$$.
- If the actual value is below the prediction (we over-forecast), the error is multiplied by
  $$1 - \tau$$.
- For $$\tau = 0.9$$, under-forecasting is penalised nine times more than over-forecasting, so the
  model learns to aim high, exactly at the level exceeded 10 percent of the time.

### Why the pinball loss finds the quantile

The expected loss for a constant prediction $$q$$ is

$$
R(q) = \tau \int_{q}^{\infty} (y - q)\, f(y)\, dy + (1 - \tau) \int_{-\infty}^{q} (q - y)\, f(y)\, dy
$$

Differentiating and setting to zero gives

$$
R'(q) = -\tau \big(1 - F(q)\big) + (1 - \tau) F(q) = F(q) - \tau = 0
\quad\Longrightarrow\quad F(q) = \tau
$$

In plain English: the prediction that minimises the pinball loss on average is precisely the
$$\tau$$ quantile. This is the same structure as the newsvendor result, with $$\tau$$ playing the role
of the critical ratio. That is not a coincidence: choosing a quantile *is* choosing how to trade off
the two kinds of error.

### Coverage and interval width

For an interval $$[\hat{q}_{\text{lo}}, \hat{q}_{\text{hi}}]$$ that should contain the outcome with
probability $$1 - \alpha$$, two numbers tell us whether it is useful:

$$
\text{Coverage} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}\big[\hat{q}_{\text{lo},i} \le y_i \le \hat{q}_{\text{hi},i}\big]
\qquad
\text{Width} = \frac{1}{n} \sum_{i=1}^{n} \big(\hat{q}_{\text{hi},i} - \hat{q}_{\text{lo},i}\big)
$$

Coverage should be close to $$1 - \alpha$$. Among intervals with correct coverage, narrower is
better. An interval from zero to infinity has perfect coverage and no value.

### Conformal calibration

Quantile models are often over-confident on new data. **Conformalized quantile regression (CQR)**
fixes this with a held-out calibration set of size $$n$$. For each calibration point compute the
conformity score

$$
E_i = \max\big(\hat{q}_{\text{lo}}(x_i) - y_i,\; y_i - \hat{q}_{\text{hi}}(x_i)\big)
$$

In plain English: how far outside the interval the true value fell (positive), or how far inside it
was (negative). Then take the adjusted quantile of these scores

$$
\hat{Q} = \text{Quantile}\!\left(E_1, \dots, E_n;\; \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n}\right)
$$

and widen (or narrow, if $$\hat{Q}$$ is negative) every future interval by it:

$$
\big[\hat{q}_{\text{lo}}(x) - \hat{Q},\; \hat{q}_{\text{hi}}(x) + \hat{Q}\big]
$$

The result has a finite-sample guarantee: if calibration and future data are exchangeable, coverage
is at least $$1 - \alpha$$. For time series that assumption is only approximately true, so I
recalibrate on a rolling recent window.

### Avoiding crossing quantiles

Separately trained quantile models can occasionally produce P90 below P50. The simplest fix is to
sort the predicted quantiles for each row, which never makes the pinball loss worse.

## Implementation in Python

### Features

```python
# forecasting/features.py
import pandas as pd


def make_features(df: pd.DataFrame, target: str = "volume") -> pd.DataFrame:
    """df: one row per (site, date), sorted by date within site."""
    out = df.copy()
    g = out.groupby("site")[target]
    for lag in (7, 14, 28):
        out[f"lag_{lag}"] = g.shift(lag)
    out["roll_mean_28"] = g.transform(lambda s: s.shift(7).rolling(28, min_periods=14).mean())
    out["roll_std_28"] = g.transform(lambda s: s.shift(7).rolling(28, min_periods=14).std())
    out["dow"] = out["date"].dt.dayofweek
    out["week"] = out["date"].dt.isocalendar().week.astype(int)
    out["month"] = out["date"].dt.month
    out["site"] = out["site"].astype("category")
    return out
```

All lags start at 7 days because the forecast is made a week ahead. Using `lag_1` here would be a
leak: yesterday's volume is not known when the plan is made.

### Quantile models

```python
# forecasting/quantile_model.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


@dataclass
class QuantileForecaster:
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    params: dict = field(default_factory=lambda: dict(n_estimators=600, learning_rate=0.03,
                                                      num_leaves=31, min_child_samples=40))
    models: dict[float, LGBMRegressor] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileForecaster":
        for q in self.quantiles:
            self.models[q] = LGBMRegressor(objective="quantile", alpha=q, **self.params).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = np.column_stack([self.models[q].predict(X) for q in self.quantiles])
        preds = np.sort(preds, axis=1)  # remove quantile crossing
        return pd.DataFrame(preds, index=X.index, columns=[f"q{int(q * 100)}" for q in self.quantiles])
```

### Evaluation and calibration

```python
# forecasting/evaluation.py
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_pinball_loss


def pinball_report(y, preds, quantiles) -> dict[str, float]:
    return {f"pinball_q{int(q * 100)}": mean_pinball_loss(y, preds[f"q{int(q * 100)}"], alpha=q)
            for q in quantiles}


def coverage(y, lo, hi) -> float:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


class ConformalAdjuster:
    """Conformalized quantile regression on a recent calibration window."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.q_hat = 0.0

    def fit(self, y_cal, lo_cal, hi_cal) -> "ConformalAdjuster":
        y, lo, hi = map(np.asarray, (y_cal, lo_cal, hi_cal))
        scores = np.maximum(lo - y, y - hi)
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level, method="higher"))
        return self

    def transform(self, lo, hi):
        return np.asarray(lo) - self.q_hat, np.asarray(hi) + self.q_hat
```

### Putting it together

```python
from forecasting.evaluation import ConformalAdjuster, coverage, pinball_report
from forecasting.features import make_features
from forecasting.quantile_model import QuantileForecaster

data = make_features(raw).dropna(subset=["lag_28", "roll_mean_28"])
feature_cols = ["lag_7", "lag_14", "lag_28", "roll_mean_28", "roll_std_28", "dow", "week", "month", "site"]

train = data[data["date"] < "2026-01-01"]
calib = data[(data["date"] >= "2026-01-01") & (data["date"] < "2026-03-01")]
test = data[data["date"] >= "2026-03-01"]

model = QuantileForecaster().fit(train[feature_cols], train["volume"])

cal_pred = model.predict(calib[feature_cols])
adjuster = ConformalAdjuster(alpha=0.2).fit(calib["volume"], cal_pred["q10"], cal_pred["q90"])

test_pred = model.predict(test[feature_cols])
lo, hi = adjuster.transform(test_pred["q10"], test_pred["q90"])

print(pinball_report(test["volume"], test_pred, model.quantiles))
print("raw coverage       ", coverage(test["volume"], test_pred["q10"], test_pred["q90"]))
print("calibrated coverage", coverage(test["volume"], lo, hi))
```

The split is by time, never random. A random split lets the model see the future of each site, and
the evaluation will look much better than reality.

### From quantiles to a decision

```python
import numpy as np


def quantile_at(preds_row: dict[float, float], tau: float) -> float:
    """Linear interpolation between predicted quantiles."""
    levels = np.array(sorted(preds_row))
    values = np.array([preds_row[q] for q in levels])
    return float(np.interp(tau, levels, values))


def staff_needed(volume: float, parcels_per_shift: float) -> int:
    return int(np.ceil(volume / parcels_per_shift))
```

## A worked example: staffing a parcel sorting hub

A regional parcel hub plans sorting shifts one week ahead. Each sorting shift handles about 1,000
parcels. The costs, agreed with operations:

- An extra shift booked but not needed costs about 2,500 SEK (paid but idle). That is the overage cost
  $$C_o$$.
- A missing shift means overtime and late-delivery penalties, about 6,000 SEK per 1,000 parcels. That
  is the underage cost $$C_u$$.

The right quantile to staff for is

$$
\tau^{*} = \frac{C_u}{C_u + C_o} = \frac{6{,}000}{8{,}500} \approx 0.71
$$

For next Monday the calibrated forecast is P10 = 10,900, P50 = 12,000, P90 = 13,600. Interpolating
between P50 and P90 at 0.71 gives about

$$
12{,}000 + \frac{0.71 - 0.50}{0.90 - 0.50} \times (13{,}600 - 12{,}000) \approx 12{,}840
$$

parcels, so the hub books 13 shifts rather than the 12 the point forecast suggests. On a quiet Tuesday
with a narrow range (P50 = 9,000, P90 = 9,500) the same rule adds almost nothing, and the planner does
not over-staff out of habit.

Over a quarter, this kind of policy typically reduces both overtime and idle hours at the same time,
which a fixed "plus 15 percent" rule cannot do. It also gives management a single, explainable lever:
if the cost of late deliveries goes up, $$\tau^{*}$$ goes up, and the staffing plan responds
automatically.

## Pitfalls and when not to use it

- **Random train/test splits.** Always split by time and forecast with realistic lead time.
- **Features that are not known at forecast time.** Actual weather, actual promotions, yesterday's
  volume when planning a week ahead. Use forecasts of those drivers, or leave them out.
- **Ignoring calibration drift.** Peak season changes the error distribution. Recalibrate on a
  rolling window and monitor coverage monthly.
- **Too many quantiles too early.** Start with P10, P50 and P90. Add more only if a decision needs
  them.
- **Very short history.** With less than a year of data, seasonality cannot be learned. A simple
  statistical model with seasonal priors may do better than gradient boosting.
- **When the decision does not depend on uncertainty.** If the cost of over and under is the same and
  capacity is flexible on the day, a good point forecast may be enough.

## Measuring success in production

- Pinball loss per quantile, compared with a seasonal-naive baseline.
- Empirical coverage of the P10 to P90 interval (target about 80 percent), by site and by month.
- Average interval width, which should shrink as the model improves.
- Business outcomes: overtime hours, idle hours, on-time delivery rate.

## Checklist

- Define the forecast horizon and use only features known at that horizon.
- Train separate quantile models with the pinball loss, then sort to remove crossings.
- Evaluate with pinball loss and coverage, not only MAPE.
- Calibrate intervals with a recent hold-out window.
- Choose the operating quantile from the cost ratio.
- Monitor coverage and recalibrate regularly.

## Final thought

Moving from point forecasts to calibrated ranges is one of the most practical upgrades a planning team
can make. It does not need exotic models, just the right loss function, an honest calibration step
and a clear link to costs. If you are planning capacity, staffing or stock under uncertainty, I would
be glad to help. See my [forecasting and predictive analytics service](/services/#forecasting-predictive-analytics)
or [contact me](/contact/).
