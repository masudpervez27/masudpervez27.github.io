---
title: "The Newsvendor Model in Practice: How Much to Order When Demand Is Uncertain"
date: 2026-04-05T10:00:00+02:00
excerpt: "Ordering the forecast is almost never the profit-maximising choice. A step-by-step derivation of the critical fractile, how to combine it with machine learning forecasts, a modular Python implementation and a worked example for a bakery chain."
categories:
  - Decision Science
  - Forecasting
tags:
  - Newsvendor
  - Inventory Optimization
  - Probabilistic Forecasting
  - Operations Research
  - SciPy
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "boxes-stacked"
---

"How many should we make tomorrow?" is one of the most common questions in operations, and one of
the most often answered with a single number from a forecast. The problem is that a forecast tells
you what is *likely* to happen, not what you should *do*. When running out and having leftovers cost
different amounts, the best quantity is not the forecast. The newsvendor model tells you exactly how
far to move away from it, and in my experience it is one of the highest-return pieces of statistics
a business can apply.

## The business problem

Any product that is ordered or produced before demand is known, and loses value if it is not sold
within a period, has this problem:

- Fresh food, bakery and ready meals.
- Printed media, event tickets and seasonal goods.
- Staff scheduled for a shift, or delivery capacity booked for a day.
- Cloud capacity reserved in advance.

Order too much and you pay for waste. Order too little and you lose the margin on sales you could
have made, and sometimes the customer as well. Most companies handle this with a rule of thumb ("the
forecast plus 10 percent") applied the same way to every product. Because the right buffer depends on
the margin and on how uncertain demand is, a single rule over-orders some products and under-orders
others at the same time.

## The idea in plain English

Think about the very last unit you might order. It will either be sold or not.

- If it sells, you earn its margin. Missing that sale would have cost you the margin. Call that the
  **underage cost**.
- If it does not sell, you lose what you paid for it, minus anything you get back. Call that the
  **overage cost**.

You should order that last unit if the chance it sells, times the underage cost, is larger than the
chance it does not sell, times the overage cost. Keep adding units until the two balance. The
balance point turns out to be a specific **percentile** of the demand distribution, and that
percentile depends only on the two costs.

High margin and cheap waste? Order high in the distribution. Thin margin and expensive waste? Order
low.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Sales history<small>corrected for sell-outs</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Probabilistic forecast<small>mean and spread, or quantiles</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Unit economics<small>price, cost, salvage</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">Critical ratio<small>underage vs overage cost</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-ok">Order quantity<small>forecast quantile at the ratio</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Track waste and sell-outs</div>
  </div>
  <p class="dg-caption">Turning a demand forecast into a profit-maximising order.</p>
</div>

## How it works: the math, step by step

### Setting up

Let:

- Selling price per unit be $$p$$.
- Cost per unit be $$c$$.
- Salvage value per unsold unit be $$s$$, with $$s < c$$ (what you recover from leftovers, possibly
  zero).
- Demand be a random variable $$D$$ with density $$f(x)$$ and cumulative distribution $$F(x)$$.
- Order quantity be $$Q$$.

The two unit costs are

$$
C_u = p - c \qquad\qquad C_o = c - s
$$

In plain English: $$C_u$$ is the margin lost on each unit of demand you could not serve, and
$$C_o$$ is the money lost on each unit you made but could not sell.

### Profit for a given order

Units sold are $$\min(Q, D)$$ and leftovers are $$(Q - D)^{+} = \max(Q - D, 0)$$. Profit is

$$
\Pi(Q, D) = p \min(Q, D) + s (Q - D)^{+} - c Q
$$

Using $$\min(Q, D) = Q - (Q - D)^{+}$$ this simplifies to

$$
\Pi(Q, D) = (p - c)\,Q - (p - s)\,(Q - D)^{+}
$$

Read it as: you earn the full margin on everything you ordered, and then give back price minus
salvage on every unit left over.

### Expected cost

It is often clearer to minimise the expected cost of mismatch:

$$
G(Q) = C_o \int_{0}^{Q} (Q - x)\, f(x)\, dx \;+\; C_u \int_{Q}^{\infty} (x - Q)\, f(x)\, dx
$$

Line by line:

- The first integral covers all demand levels below $$Q$$. For each one, $$Q - x$$ units are left
  over, each costing $$C_o$$, weighted by how likely that demand level is.
- The second integral covers all demand levels above $$Q$$. For each, $$x - Q$$ units of demand are
  missed, each costing $$C_u$$.

### Finding the best quantity

Differentiate with respect to $$Q$$. By the Leibniz rule, the boundary terms vanish because the
integrands are zero at $$x = Q$$, leaving

$$
G'(Q) = C_o \int_{0}^{Q} f(x)\, dx - C_u \int_{Q}^{\infty} f(x)\, dx = C_o F(Q) - C_u \big(1 - F(Q)\big)
$$

In plain English: raising the order by one unit costs $$C_o$$ in the cases where demand is already
below $$Q$$ (probability $$F(Q)$$), and saves $$C_u$$ in the cases where demand is above $$Q$$
(probability $$1 - F(Q)$$).

Set the derivative to zero and solve:

$$
F(Q^{*}) = \frac{C_u}{C_u + C_o}
$$

This ratio is called the **critical fractile** or critical ratio. The optimal order is

$$
Q^{*} = F^{-1}\!\left(\frac{C_u}{C_u + C_o}\right)
$$

Read it as: order the quantity that demand stays below with probability equal to the critical
ratio. Since $$G''(Q) = (C_o + C_u) f(Q) \ge 0$$, this is a minimum, not a maximum.

### The normal case

If demand is approximately normal with mean $$\mu$$ and standard deviation $$\sigma$$, then

$$
Q^{*} = \mu + z^{*} \sigma, \qquad z^{*} = \Phi^{-1}\!\left(\frac{C_u}{C_u + C_o}\right)
$$

where $$\Phi$$ is the standard normal distribution. The expected leftover has a closed form:

$$
E\big[(Q - D)^{+}\big] = \sigma \big(z \, \Phi(z) + \varphi(z)\big), \qquad z = \frac{Q - \mu}{\sigma}
$$

with $$\varphi$$ the standard normal density. That lets us compute expected profit exactly for any
$$Q$$ and compare policies.

### Beyond the normal distribution

Real demand is often skewed, cannot be negative, and for slow sellers is a small whole number. Two
practical options:

- **Negative binomial** for counts with extra variance. With mean $$\mu$$ and dispersion $$\theta$$,
  the variance is $$\mu + \mu^2 / \theta$$.
- **Empirical quantiles** from a probabilistic forecast or from historical residuals. Then
  $$Q^{*}$$ is simply the forecast's quantile at the critical ratio. This is the option I use most,
  because it works with any forecasting model.

## Implementation in Python

```python
# newsvendor.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class UnitEconomics:
    price: float
    cost: float
    salvage: float = 0.0

    def __post_init__(self) -> None:
        if not (self.salvage < self.cost < self.price):
            raise ValueError("Expected salvage < cost < price")

    @property
    def underage(self) -> float:
        return self.price - self.cost

    @property
    def overage(self) -> float:
        return self.cost - self.salvage

    @property
    def critical_ratio(self) -> float:
        return self.underage / (self.underage + self.overage)


def q_normal(mu: float, sigma: float, econ: UnitEconomics) -> float:
    return float(mu + stats.norm.ppf(econ.critical_ratio) * sigma)


def q_negative_binomial(mu: float, theta: float, econ: UnitEconomics) -> int:
    var = mu + mu**2 / theta
    p = mu / var
    n = mu**2 / (var - mu)
    return int(stats.nbinom.ppf(econ.critical_ratio, n, p))


def q_empirical(demand_samples: np.ndarray, econ: UnitEconomics) -> float:
    """Works with any sample of plausible demand: history, bootstrap, or forecast draws."""
    return float(np.quantile(demand_samples, econ.critical_ratio, method="higher"))


def expected_profit_normal(q: float, mu: float, sigma: float, econ: UnitEconomics) -> float:
    z = (q - mu) / sigma
    leftover = sigma * (z * stats.norm.cdf(z) + stats.norm.pdf(z))
    return (econ.price - econ.cost) * q - (econ.price - econ.salvage) * leftover


def expected_profit_simulated(q: float, demand_samples: np.ndarray, econ: UnitEconomics) -> float:
    sold = np.minimum(q, demand_samples)
    left = np.maximum(q - demand_samples, 0)
    return float(np.mean(econ.price * sold + econ.salvage * left - econ.cost * q))
```

### Plugging in a machine learning forecast

In practice $$\mu$$ and $$\sigma$$ differ per product, store and day. A forecasting model provides
them, or provides the quantile directly. A small planning function then turns forecasts into orders:

```python
import pandas as pd

from newsvendor import UnitEconomics, q_normal


def plan_orders(forecast: pd.DataFrame, econ_by_product: dict[str, UnitEconomics]) -> pd.DataFrame:
    """forecast columns: store, product, date, mu, sigma"""
    def row_q(r) -> int:
        econ = econ_by_product[r["product"]]
        return max(0, round(q_normal(r["mu"], r["sigma"], econ)))

    out = forecast.copy()
    out["critical_ratio"] = out["product"].map(lambda p: econ_by_product[p].critical_ratio)
    out["order_qty"] = out.apply(row_q, axis=1)
    return out
```

If your forecasting model outputs quantiles directly (see my post on
[probabilistic demand forecasting](/posts/2026-05-14-probabilistic-demand-forecasting-quantile-regression-logistics/)),
you can skip the normal assumption and read off the quantile closest to the critical ratio.

## A worked example: a bakery chain

A bakery chain sells a sourdough loaf in 30 stores.

- Price $$p = 40$$ SEK, cost $$c = 15$$ SEK, and unsold loaves go to a discount partner for
  $$s = 3$$ SEK.
- So $$C_u = 25$$ and $$C_o = 12$$, and the critical ratio is $$25 / 37 \approx 0.676$$.
- For a typical store on a weekday, the forecast is $$\mu = 200$$ with $$\sigma = 40$$.

The optimal order is

$$
z^{*} = \Phi^{-1}(0.676) \approx 0.456, \qquad Q^{*} = 200 + 0.456 \times 40 \approx 218
$$

Compare expected profit for ordering the forecast (200) and ordering 218:

- At $$Q = 200$$: $$z = 0$$, expected leftovers are $$40 \times 0.399 \approx 16.0$$, so expected
  profit is $$25 \times 200 - 37 \times 16.0 \approx 4{,}410$$ SEK.
- At $$Q = 218$$: $$z = 0.45$$, expected leftovers are
  $$40 \times (0.45 \times 0.674 + 0.360) \approx 26.5$$, so expected profit is
  $$25 \times 218 - 37 \times 26.5 \approx 4{,}468$$ SEK.

That is about 58 SEK more per store per day for one product, from changing a number. Across 30 stores
and a year, it is roughly 600,000 SEK, before applying the same logic to the rest of the range.

Notice what happens for a product with a thinner margin. A pastry with $$p = 30$$, $$c = 18$$,
$$s = 0$$ has a critical ratio of $$12 / 30 = 0.40$$, so the right order is *below* the forecast. The
"forecast plus 10 percent" rule was over-ordering this item every day.

## Pitfalls and when not to use it

- **Using sales as demand.** When a product sells out, sales understate demand. Training on raw sales
  makes the forecast too low, which makes orders too low, which causes more sell-outs. Flag stock-out
  days and treat them as censored observations.
- **Ignoring the hidden costs.** A stock-out may cost more than the lost margin if customers switch
  stores. Add a goodwill penalty to $$C_u$$ when you can estimate it.
- **Assuming normality for slow movers.** For items that sell 0 to 5 per day, use a count
  distribution or empirical quantiles.
- **Multi-period products.** If leftovers can be sold tomorrow at full price, this is an inventory
  problem with carry-over, not a single-period newsvendor. Use a reorder-point or base-stock policy
  instead (see my post on [safety stock and reorder points](/posts/2026-06-18-safety-stock-reorder-points-uncertain-lead-times/)).
- **Shared constraints.** If oven or shelf capacity limits the total, solve the products jointly with
  a Lagrange multiplier on capacity, rather than one by one.

## Measuring success in production

- Waste rate (unsold share) and stock-out rate per product and store.
- Realised margin after waste, against a control group still using the old rule.
- Forecast calibration: does demand fall below the P67 about 67 percent of the time?
- Adoption: how often store managers override the suggested order, and why.

## Checklist

- Get price, cost and salvage per product from finance, and store them as data.
- Compute the critical ratio per product, not one ratio for the business.
- Produce a probabilistic forecast (mean and spread, or quantiles).
- Correct for stock-outs in the training data.
- Run a controlled pilot in a subset of stores before rolling out.
- Report profit, not forecast accuracy, to the business owner.

## Final thought

The newsvendor model is more than seventy years old, and it still beats most rules of thumb because
it asks the right question: not "what will demand be?" but "what should we do given what demand might
be?". Combined with a modern probabilistic forecast, it turns forecasting from a reporting exercise
into a margin lever. If you want to apply this to your own range, see my
[forecasting, optimisation and decision analytics service](/services/#forecasting-decision-analytics) or
[contact me](/contact/).
