---
title: "Reliable Feature Pipelines with PySpark: Point-in-Time Correctness, Idempotent Jobs and Drift Monitoring"
date: 2026-02-20T10:00:00+01:00
excerpt: "Most models that fail after launch fail because of the data pipeline, not the algorithm. How to build Spark feature pipelines that do not leak the future, can be rerun safely, and warn you when the data changes, with the math behind each check."
categories:
  - MLOps
  - Data Engineering
tags:
  - Apache Spark
  - PySpark
  - Feature Engineering
  - Data Quality
  - Drift Detection
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "network-wired"
---

When a model looks excellent in the notebook and disappointing in production, I now check the
feature pipeline before I check the model. In most cases the cause is one of three things: a feature
that accidentally used information from the future, a job that produced different numbers when it
was rerun, or input data that slowly changed while nobody was watching. This post covers how I design
Spark feature pipelines to prevent all three, with the math behind each check and modular PySpark
code you can adapt.

## The business problem

A feature pipeline turns raw events (orders, logins, sensor readings, page views) into the table a
model learns from and scores on. When it goes wrong, the symptoms are expensive and slow to diagnose:

- The model is approved on validation results that were too good, because of leakage. The business
  plans around numbers that will never appear in reality.
- A failed job is rerun and silently duplicates a day of data. Reports and model scores shift and
  nobody can say why.
- An upstream system changes a unit or a category code. The model keeps scoring, the scores become
  meaningless, and the problem is noticed weeks later in the business results.

Each of these costs far more in lost trust than in compute. A well designed pipeline makes them
either impossible or loud.

## The idea in plain English

Three rules carry most of the weight:

1. **Only use the past.** When you compute a feature for a given moment, use only events that had
   already happened at that moment. Exactly as the model would see them in production.
2. **Same input, same output.** Running the job twice for the same day must give the same result as
   running it once. Then retries and backfills are safe.
3. **Watch the inputs, not just the outputs.** Compare each day's feature distributions with a
   reference period, and raise an alert when they move more than expected.

Everything else, like partitioning, caching and cluster sizing, is about making these three rules
fast and cheap.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Raw events<small>data lake</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">Point-in-time join<small>only events before the snapshot</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Window and decay features</div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Quality checks<small>unique keys, null rates</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-warn">Drift check<small>PSI against a reference</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-ok">Partition overwrite<small>feature table for training and scoring</small></div>
  </div>
  <p class="dg-caption">One daily job, safe to rerun, used for both the training backfill and live scoring.</p>
</div>

## How it works: the math, step by step

### Point-in-time correctness

Let each entity (a customer, a machine, a store) be $$e$$, and let it produce events $$(t_j, v_j)$$
with timestamp $$t_j$$ and value $$v_j$$. A rolling-window feature at time $$t$$ with window length
$$w$$ is

$$
x_e^{\text{sum}}(t) = \sum_{j \,:\, t - w \le t_j < t} v_j
$$

In plain English: add up the values of all events that happened in the window before $$t$$. The
strict inequality $$t_j < t$$ is the important part. The event happening *at* time $$t$$ is what we
are trying to predict or react to, so it must not be part of its own features.

The label is defined over a future window of length $$h$$:

$$
y_e(t) = \mathbf{1}\left[\text{target event occurs in } [t, \, t + h)\right]
$$

In plain English: did the thing we care about happen in the next $$h$$ days after the moment we made
the prediction. Features look backwards from $$t$$; labels look forwards from $$t$$. If any feature
window overlaps the label window, the model is learning from the answer.

### Recency-weighted features

A plain window treats an event 29 days ago the same as one yesterday. An exponentially decayed sum
weights recent events more:

$$
x_e^{\text{decay}}(t) = \sum_{j \,:\, t_j < t} v_j \, e^{-\lambda (t - t_j)},
\qquad \lambda = \frac{\ln 2}{T_{1/2}}
$$

Read it as: each past event counts fully when it is fresh, and its weight halves every
$$T_{1/2}$$ days. A half-life of 7 days means an event from two weeks ago counts one quarter.

### Idempotency

Let $$f$$ be the job that computes features for partition $$d$$ (a day). The job is idempotent if

$$
f(f(S, d), d) = f(S, d)
$$

where $$S$$ is the state of the feature table. In plain English: running the job for day $$d$$ a
second time leaves the table exactly as it was after the first run. In Spark this is achieved by
**overwriting the partition** for day $$d$$ rather than appending to it.

### Population Stability Index (PSI)

To detect drift in a feature, split its reference distribution into $$B$$ bins (usually deciles).
Let $$r_b$$ be the share of reference rows in bin $$b$$ and $$a_b$$ the share of today's rows in the
same bin. Then

$$
\text{PSI} = \sum_{b=1}^{B} (a_b - r_b) \, \ln\!\left(\frac{a_b}{r_b}\right)
$$

Line by line:

- $$a_b - r_b$$ says how much the share in bin $$b$$ moved.
- $$\ln(a_b / r_b)$$ says how large that move is relative to the original share, and has the same
  sign, so every term is zero or positive.
- Summing over bins gives one number per feature.

Common rules of thumb: below 0.1 is stable, 0.1 to 0.25 deserves a look, above 0.25 is a significant
shift. PSI is a symmetric form of the Kullback-Leibler divergence, which is why it behaves well as a
distance.

### Kolmogorov-Smirnov statistic

A second check that does not need bins is the largest gap between the two cumulative distributions:

$$
D = \sup_{x} \left| F_{\text{ref}}(x) - F_{\text{today}}(x) \right|
$$

In plain English: walk along all possible values and find the point where the two distributions
disagree most about "what share of values are below this". With very large samples almost any
difference becomes statistically significant, so I use $$D$$ itself as an effect size with a fixed
alert level rather than relying on the p-value.

## Implementation in Python

### Structure

I keep each job in four small layers, so every part can be tested without a cluster:

```
features/
  config.py        # windows, paths, thresholds (no logic)
  transforms.py    # pure DataFrame -> DataFrame functions
  quality.py       # data quality and drift checks
  job.py           # read, transform, check, write for one partition
```

### Configuration

```python
# features/config.py
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConfig:
    entity_col: str = "customer_id"
    ts_col: str = "event_ts"
    value_col: str = "amount"
    windows_days: tuple[int, ...] = (7, 30, 90)
    half_life_days: float = 14.0
    events_path: str = "s3://lake/events/"
    output_path: str = "s3://lake/features/customer_daily/"
    psi_alert: float = 0.25
    null_rate_alert: float = 0.02
    key_cols: tuple[str, ...] = field(default=("customer_id", "as_of_date"))
```

### Transforms

```python
# features/transforms.py
import math

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from features.config import FeatureConfig

DAY = 86_400


def snapshot_rows(entities: DataFrame, as_of_date: str) -> DataFrame:
    """One row per entity at the start of the scoring day."""
    return entities.select("customer_id").distinct().withColumn(
        "as_of_ts", F.to_timestamp(F.lit(as_of_date))
    ).withColumn("as_of_date", F.lit(as_of_date).cast("date"))


def window_features(events: DataFrame, snapshots: DataFrame, cfg: FeatureConfig) -> DataFrame:
    """Rolling sums and counts using only events strictly before as_of_ts."""
    e, s = events.alias("e"), snapshots.alias("s")
    max_window = max(cfg.windows_days) * DAY
    joined = s.join(
        e,
        on=(F.col(f"e.{cfg.entity_col}") == F.col(f"s.{cfg.entity_col}"))
        & (F.col(f"e.{cfg.ts_col}") < F.col("s.as_of_ts"))
        & (F.col(f"e.{cfg.ts_col}") >= F.col("s.as_of_ts") - F.expr(f"INTERVAL {max_window} SECONDS")),
        how="left",
    )
    age_s = F.col("s.as_of_ts").cast("long") - F.col(f"e.{cfg.ts_col}").cast("long")

    aggs = []
    for d in cfg.windows_days:
        in_window = age_s <= d * DAY
        aggs.append(F.sum(F.when(in_window, F.col(f"e.{cfg.value_col}"))).alias(f"sum_{d}d"))
        aggs.append(F.count(F.when(in_window, F.lit(1))).alias(f"count_{d}d"))

    decay = math.log(2) / (cfg.half_life_days * DAY)
    aggs.append(F.sum(F.col(f"e.{cfg.value_col}") * F.exp(-decay * age_s)).alias("decayed_sum"))

    out = joined.groupBy(F.col(f"s.{cfg.entity_col}").alias(cfg.entity_col), "s.as_of_date").agg(*aggs)
    return out.fillna(0)


def ratio_features(df: DataFrame, short: int = 7, long: int = 90) -> DataFrame:
    """How the recent week compares with the longer-term average."""
    weekly_avg_long = F.col(f"sum_{long}d") / F.lit(long / short)
    return df.withColumn(
        f"trend_{short}d_vs_{long}d",
        F.when(weekly_avg_long > 0, F.col(f"sum_{short}d") / weekly_avg_long),
    )
```

The join condition is where point-in-time correctness lives. `e.event_ts < s.as_of_ts` makes it
impossible for an event at or after the snapshot time to enter a feature, no matter how the rest of
the code changes.

### Quality and drift checks

```python
# features/quality.py
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class DataQualityError(RuntimeError):
    pass


def check_unique_keys(df: DataFrame, keys: tuple[str, ...]) -> None:
    dupes = df.groupBy(*keys).count().filter("count > 1").limit(1).count()
    if dupes:
        raise DataQualityError(f"Duplicate keys on {keys}")


def check_null_rates(df: DataFrame, max_rate: float) -> dict[str, float]:
    total = df.count()
    rates = df.select([(F.sum(F.col(c).isNull().cast("int")) / total).alias(c) for c in df.columns]).first()
    bad = {c: r for c, r in rates.asDict().items() if r is not None and r > max_rate}
    if bad:
        raise DataQualityError(f"Null rate too high: {bad}")
    return rates.asDict()


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference) + eps
    a = np.histogram(current, edges)[0] / len(current) + eps
    return float(np.sum((a - r) * np.log(a / r)))


def drift_report(reference: DataFrame, current: DataFrame, cols: list[str],
                 sample_rows: int = 200_000) -> dict[str, float]:
    """PSI per feature on bounded samples, so the check stays cheap."""
    def sample(df: DataFrame, c: str) -> np.ndarray:
        n = df.count()
        frac = min(1.0, sample_rows / max(n, 1))
        rows = df.select(c).dropna().sample(fraction=frac, seed=11).collect()
        return np.array([row[0] for row in rows], dtype=float)

    return {c: psi(sample(reference, c), sample(current, c)) for c in cols}
```

### The job

```python
# features/job.py
from pyspark.sql import SparkSession

from features import quality, transforms
from features.config import FeatureConfig


def run(spark: SparkSession, as_of_date: str, cfg: FeatureConfig = FeatureConfig()) -> None:
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    events = spark.read.parquet(cfg.events_path)
    snapshots = transforms.snapshot_rows(events, as_of_date)
    features = transforms.ratio_features(transforms.window_features(events, snapshots, cfg))

    quality.check_unique_keys(features, cfg.key_cols)
    quality.check_null_rates(features.drop("trend_7d_vs_90d"), cfg.null_rate_alert)

    reference = spark.read.parquet(cfg.output_path).filter(
        f"as_of_date BETWEEN date_sub(to_date('{as_of_date}'), 120) AND date_sub(to_date('{as_of_date}'), 30)"
    )
    drift = quality.drift_report(reference, features, ["sum_30d", "count_30d", "decayed_sum"])
    alerts = {k: v for k, v in drift.items() if v > cfg.psi_alert}
    if alerts:
        print(f"DRIFT WARNING {as_of_date}: {alerts}")  # send to your alerting channel

    (features.write.mode("overwrite")
        .partitionBy("as_of_date")
        .parquet(cfg.output_path))
```

`partitionOverwriteMode=dynamic` with `mode("overwrite")` replaces only the partitions present in the
DataFrame, which is today's date. That single setting is what makes the job idempotent: run it once
or five times, the table ends up the same.

### Testing a transform locally

```python
# tests/test_point_in_time.py
from datetime import datetime

from features.config import FeatureConfig
from features.transforms import snapshot_rows, window_features


def test_event_at_snapshot_time_is_excluded(spark):
    events = spark.createDataFrame(
        [("c1", datetime(2026, 2, 19, 12), 10.0),
         ("c1", datetime(2026, 2, 20, 0), 999.0)],   # exactly at as_of_ts: must not count
        ["customer_id", "event_ts", "amount"],
    )
    snaps = snapshot_rows(events, "2026-02-20")
    row = window_features(events, snaps, FeatureConfig()).first()
    assert row["sum_7d"] == 10.0
```

A test like this takes seconds on a laptop and protects the one property that, if broken, makes every
validation number meaningless.

## A worked example: daily customer features for a subscription service

A subscription business with 3 million customers wants to score cancellation risk every morning, and
to retrain monthly on two years of history.

- **Backfill.** The same `run(spark, as_of_date)` is called for each of the 730 historical days. The
  output is a training table where each row shows exactly what was known that morning. No separate
  "training version" of the logic exists, so training and scoring cannot drift apart.
- **Daily run.** The scheduler calls the job for today at 03:00. If it fails halfway, it is simply
  rerun. Dynamic partition overwrite guarantees no duplicates.
- **Drift catch.** One Monday, the PSI of `sum_30d` jumps to 0.6. Investigation shows an upstream
  system started recording amounts in öre instead of kronor. The alert fires before the morning
  scoring is used by the marketing team, the source is fixed, and the partition is recomputed.

Without the check, that unit change would have pushed every customer into "high value" and wasted a
retention campaign on the wrong people.

## Pitfalls and when not to use it

- **Using `current_timestamp()` inside transforms.** It makes reruns non-reproducible. Always pass
  the `as_of_date` in.
- **Joining on "latest" dimension tables.** If a customer's segment is joined from today's snapshot
  while building historical rows, you leak the future. Use dated snapshots or slowly changing
  dimension tables with validity ranges.
- **Skewed keys.** A handful of very active entities can make one Spark task run for hours. Check the
  key distribution and use salting or pre-aggregation for the heavy keys.
- **PSI on tiny samples.** With few rows per bin, PSI is noisy. Use a minimum sample size or fewer
  bins.
- **Spark for small data.** If the data fits comfortably in memory on one machine, pandas or Polars
  with the same design principles is simpler and cheaper. The principles matter more than the engine.

## Measuring success in production

- Job success rate and runtime per day, and number of manual reruns.
- Gap between offline validation metrics and live performance (leakage shows up here).
- Number of drift alerts per month and the share that pointed to real issues.
- Time from an upstream change to its detection.
- Compute cost per daily run.

## Checklist

- Every transform takes `as_of_date` as an argument and never reads the clock.
- Join conditions enforce `event_ts < as_of_ts`.
- Writes use dynamic partition overwrite.
- Unique-key and null-rate checks run before the write.
- PSI is computed for key features against a fixed reference window.
- A unit test proves that events at the snapshot time are excluded.
- The same code builds both the training backfill and the daily scoring table.

## Final thought

Good feature pipelines are not glamorous, but they decide whether a model can be trusted after its
first month. Building them around point-in-time correctness, idempotency and drift monitoring has
saved me far more time than any modelling trick. If your models behave differently in production
than they did in testing, I can help find out why. See my
[production MLOps service](/services/#production-mlops) or [contact me](/contact/).
