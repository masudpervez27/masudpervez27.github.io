---
title: "Safety Stock and Reorder Points When Both Demand and Lead Time Are Uncertain"
date: 2026-06-18T10:00:00+02:00
excerpt: "Most safety stock formulas in spreadsheets ignore the fact that suppliers are late. A step-by-step derivation of safety stock under uncertain demand and lead time, the difference between cycle service and fill rate, a simulation check in Python, and a spare-parts warehouse example."
categories:
  - Forecasting
  - Logistics
tags:
  - Inventory Optimization
  - Safety Stock
  - Supply Chain
  - Monte Carlo Simulation
  - SciPy
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "warehouse"
---

When I review an inventory setup, the safety stock formula is usually the first thing I check, and
very often it only accounts for demand variability. That quietly assumes every delivery arrives
exactly on time. In real supply chains suppliers are late, ships wait at ports and trucks miss slots,
and that variability can matter more than demand itself. This post derives safety stock properly when
both demand and lead time are uncertain, explains the two service-level definitions that are often
confused, and shows how to verify the result with a simulation in Python.

## The business problem

Every warehouse and distribution centre faces the same trade-off for items that are replenished
continuously:

- Too little stock means backorders, lost orders, idle production lines and emergency freight.
- Too much stock ties up working capital, uses warehouse space and increases the risk of
  obsolescence.

Safety stock is the buffer that protects against uncertainty during the replenishment lead time. If
it is computed with the wrong formula, the business either carries far more capital than needed or,
more commonly, runs out more often than the stated service level promises. The second case is worse
because nobody notices until customers complain, and then the reaction is usually a blanket increase
in stock across all items.

## The idea in plain English

When you place a replenishment order, you need enough stock on hand to cover demand **until the
order arrives**. Two things are uncertain:

1. How much customers will take each day.
2. How many days the delivery will take.

The reorder point is the stock level at which you place the next order. It equals the demand you
*expect* during the lead time, plus a safety buffer for the cases where demand is higher or the
delivery is slower than expected. The size of that buffer depends on how much protection you want
(the service level) and how uncertain the total demand during the lead time is.

The key insight is that a late delivery multiplies the problem: every extra day of delay adds a full
day of average demand that must be covered. That is why lead-time variability often dominates.

## How it works: the math, step by step

### Demand during the lead time

Let daily demand $$d_t$$ be independent with mean $$\mu_d$$ and standard deviation $$\sigma_d$$, and
let the lead time $$L$$ (in days) be independent of demand with mean $$\mu_L$$ and standard deviation
$$\sigma_L$$. Demand during the lead time is the random sum

$$
D_L = \sum_{t=1}^{L} d_t
$$

In plain English: add up the daily demands for however many days the delivery takes. Both the
daily values and the number of days are random.

### Its mean

By the law of total expectation,

$$
E[D_L] = E\big[E[D_L \mid L]\big] = E[L \, \mu_d] = \mu_L \, \mu_d
$$

Read it as: if you knew the lead time was $$L$$ days, you would expect $$L \mu_d$$ units; averaging
over possible lead times gives average lead time times average daily demand.

### Its variance

By the law of total variance,

$$
\operatorname{Var}(D_L) = E\big[\operatorname{Var}(D_L \mid L)\big] + \operatorname{Var}\big(E[D_L \mid L]\big)
$$

Each piece in turn:

- Given $$L$$, the variance of a sum of $$L$$ independent days is $$L \sigma_d^2$$, so the first term
  is $$E[L \sigma_d^2] = \mu_L \sigma_d^2$$. This is **demand uncertainty**.
- Given $$L$$, the mean is $$L \mu_d$$, so the second term is
  $$\operatorname{Var}(L \mu_d) = \mu_d^2 \sigma_L^2$$. This is **lead-time uncertainty**.

Together:

$$
\sigma_{D_L} = \sqrt{\mu_L \, \sigma_d^2 + \mu_d^2 \, \sigma_L^2}
$$

In plain English: the uncertainty of lead-time demand has two parts. One grows with how noisy daily
demand is. The other grows with how unreliable the supplier is, multiplied by the square of average
demand. For fast-moving items with unreliable suppliers, the second part is usually the larger.

### Safety stock and reorder point for a cycle service level

The **cycle service level** $$\alpha$$ is the probability of not running out during a replenishment
cycle. If lead-time demand is approximately normal,

$$
SS = z_{\alpha} \, \sigma_{D_L}, \qquad z_{\alpha} = \Phi^{-1}(\alpha)
$$

$$
ROP = \mu_L \, \mu_d + SS
$$

Read it as: hold enough buffer to cover lead-time demand up to its $$\alpha$$ percentile, and reorder
as soon as stock on hand plus on order falls to that level. For $$\alpha = 0.95$$, $$z \approx 1.645$$.

### Fill rate is a different promise

Many businesses actually care about the **fill rate** $$\beta$$: the share of demand served directly
from stock. These are not the same. A 95 percent cycle service level means 1 cycle in 20 has *some*
shortage, but the shortage is often small, so the fill rate can be 99 percent or more.

The expected shortage per cycle under a normal approximation is

$$
ESC = \sigma_{D_L} \, G(z), \qquad G(z) = \varphi(z) - z\,\big(1 - \Phi(z)\big)
$$

where $$G(z)$$ is the standard normal loss function. With an order quantity $$Q$$ per cycle,

$$
\beta = 1 - \frac{ESC}{Q}
$$

In plain English: of the $$Q$$ units that flow through each cycle, on average $$ESC$$ units are short.
To hit a target fill rate, solve $$G(z) = (1 - \beta)\, Q / \sigma_{D_L}$$ for $$z$$. Large order
quantities need less safety stock for the same fill rate, because shortages only happen at the end of
each cycle.

### When the normal approximation fails

For slow movers, lumpy demand or skewed lead times (deliveries can be very late, but never very
early), the normal assumption underestimates the risk. Then simulate: draw lead times and daily
demands from their empirical distributions, build many samples of $$D_L$$, and set the reorder point
at the empirical $$\alpha$$ quantile.

## Implementation in Python

```python
# inventory/safety_stock.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass(frozen=True)
class ItemStats:
    mu_d: float      # mean daily demand
    sigma_d: float   # std of daily demand
    mu_l: float      # mean lead time in days
    sigma_l: float   # std of lead time in days

    @classmethod
    def from_history(cls, daily_demand: np.ndarray, lead_times: np.ndarray) -> "ItemStats":
        return cls(float(np.mean(daily_demand)), float(np.std(daily_demand, ddof=1)),
                   float(np.mean(lead_times)), float(np.std(lead_times, ddof=1)))

    @property
    def lead_time_demand_mean(self) -> float:
        return self.mu_l * self.mu_d

    @property
    def lead_time_demand_std(self) -> float:
        return float(np.sqrt(self.mu_l * self.sigma_d**2 + self.mu_d**2 * self.sigma_l**2))


def normal_loss(z: float) -> float:
    return float(stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))


def policy_cycle_service(item: ItemStats, alpha: float) -> dict[str, float]:
    z = stats.norm.ppf(alpha)
    ss = z * item.lead_time_demand_std
    return {"z": float(z), "safety_stock": float(ss), "reorder_point": item.lead_time_demand_mean + ss}


def policy_fill_rate(item: ItemStats, beta: float, order_qty: float) -> dict[str, float]:
    sigma = item.lead_time_demand_std
    target = (1 - beta) * order_qty / sigma
    z = optimize.brentq(lambda z: normal_loss(z) - target, -4, 6)
    ss = z * sigma
    return {"z": float(z), "safety_stock": float(ss), "reorder_point": item.lead_time_demand_mean + ss}


def simulate_lead_time_demand(daily_demand: np.ndarray, lead_times: np.ndarray,
                              n_sims: int = 20_000, seed: int = 0) -> np.ndarray:
    """Bootstrap: resample observed lead times and observed daily demands."""
    rng = np.random.default_rng(seed)
    lt = rng.choice(lead_times, size=n_sims, replace=True).astype(int)
    out = np.empty(n_sims)
    for i, days in enumerate(lt):
        out[i] = rng.choice(daily_demand, size=days, replace=True).sum()
    return out


def policy_simulated(daily_demand: np.ndarray, lead_times: np.ndarray, alpha: float) -> dict[str, float]:
    samples = simulate_lead_time_demand(daily_demand, lead_times)
    rop = float(np.quantile(samples, alpha))
    return {"safety_stock": rop - float(samples.mean()), "reorder_point": rop}
```

Running all three policies side by side for each item is the best sanity check I know. When the
normal formula and the simulation disagree by a lot, the item's data is telling you something
(skewed lead times, intermittent demand) that deserves a closer look.

```python
import numpy as np

from inventory.safety_stock import ItemStats, policy_cycle_service, policy_fill_rate, policy_simulated

daily = np.array(history_df.loc[history_df.sku == "PUMP-SEAL-40", "qty"])
leads = np.array(receipts_df.loc[receipts_df.sku == "PUMP-SEAL-40", "lead_time_days"])

item = ItemStats.from_history(daily, leads)
print(policy_cycle_service(item, alpha=0.95))
print(policy_fill_rate(item, beta=0.99, order_qty=400))
print(policy_simulated(daily, leads, alpha=0.95))
```

### Applying it across thousands of items

```python
import pandas as pd


def compute_policies(stats_df: pd.DataFrame, alpha_by_class: dict[str, float]) -> pd.DataFrame:
    """stats_df columns: sku, abc_class, mu_d, sigma_d, mu_l, sigma_l"""
    rows = []
    for r in stats_df.itertuples(index=False):
        item = ItemStats(r.mu_d, r.sigma_d, r.mu_l, r.sigma_l)
        policy = policy_cycle_service(item, alpha_by_class[r.abc_class])
        rows.append({"sku": r.sku, **policy, "lead_time_share": lead_time_share(item)})
    return pd.DataFrame(rows)


def lead_time_share(item: ItemStats) -> float:
    """Share of lead-time demand variance caused by supplier variability."""
    lt_var = item.mu_d**2 * item.sigma_l**2
    return lt_var / (item.mu_l * item.sigma_d**2 + lt_var)
```

The `lead_time_share` column is one I always include in the output. It tells procurement exactly
which items would benefit more from a more reliable supplier than from better forecasting.

## A worked example: a spare-parts warehouse

A distributor of industrial spare parts stocks a pump seal with:

- Average daily demand $$\mu_d = 40$$, standard deviation $$\sigma_d = 12$$.
- Average supplier lead time $$\mu_L = 6$$ days, standard deviation $$\sigma_L = 2$$ days.
- Target cycle service level 95 percent, so $$z = 1.645$$.

The two variance components are

$$
\mu_L \sigma_d^2 = 6 \times 144 = 864, \qquad \mu_d^2 \sigma_L^2 = 1{,}600 \times 4 = 6{,}400
$$

so

$$
\sigma_{D_L} = \sqrt{864 + 6{,}400} = \sqrt{7{,}264} \approx 85.2
$$

and

$$
SS = 1.645 \times 85.2 \approx 140, \qquad ROP = 6 \times 40 + 140 = 380
$$

Now compare with the common spreadsheet formula that ignores lead-time variability:

$$
SS_{\text{demand only}} = 1.645 \times \sqrt{864} \approx 1.645 \times 29.4 \approx 48
$$

The simple formula recommends 48 units of safety stock instead of 140. In practice that warehouse
would run out far more often than the promised 1 cycle in 20, because 88 percent of the uncertainty
comes from the supplier. The right response here is not only to hold more stock. It is also to show
procurement that reducing the lead-time standard deviation from 2 days to 1 day would cut the safety
stock to about $$1.645 \times \sqrt{864 + 1{,}600} \approx 82$$ units, freeing working capital
permanently. That is a negotiation argument with a number attached.

## Pitfalls and when not to use it

- **Measuring lead time from the wrong timestamps.** Use the time from order placement to the stock
  being available for picking, not to the truck arriving at the gate.
- **Mixing demand and sales.** As with forecasting, stock-outs hide demand. Correct for them.
- **Correlated demand and lead time.** In peak season both demand and delays rise together. The
  formula assumes independence, so simulate with data from the same period instead.
- **Intermittent demand.** For items with many zero-demand days, use a compound distribution or the
  simulation approach rather than the normal formula.
- **One service level for all items.** Use ABC or criticality classes. A critical spare part for a
  production line deserves 99 percent; a slow-moving accessory may need 90.
- **Static parameters.** Recompute monthly. Supplier performance changes.

## Measuring success in production

- Achieved cycle service level and fill rate per item class, against target.
- Average inventory value and inventory turns.
- Number of emergency orders and expedited shipments.
- Supplier on-time rate and lead-time standard deviation, tracked as a KPI in its own right.

## Checklist

- Measure lead time per supplier and item from real receipts.
- Use the combined formula with both demand and lead-time variance.
- Decide whether the business promise is cycle service level or fill rate.
- Validate the normal formula with a bootstrap simulation for each item class.
- Report the lead-time share of variance to procurement.
- Set service levels per ABC or criticality class and review them quarterly.

## Final thought

A correct safety stock formula is a small change with a large effect: fewer stock-outs where it
matters, less capital tied up where it does not, and a clear, quantified case for working with
suppliers on reliability. If you want an inventory policy that matches what your supply chain actually
does, see my [forecasting and predictive analytics service](/services/#forecasting-predictive-analytics)
or [get in touch](/contact/).
