---
title: "Scaling Distributed Feature Pipelines & MLOps with PySpark and Microservices"
date: 2026-02-20T10:00:00+01:00
categories:
  - MLOps
  - Big Data
tags:
  - Apache Spark
  - PySpark
  - Microservices
  - Drift Detection
  - Hadoop
header:
  teaser: /assets/images/bg-post-resize.jpg
author_profile: true
read_time: true
toc: true
toc_label: "Contents"
toc_icon: "server"
---

When training models on tens of millions of historical events across multi-terabyte data lakes, local Python pipelines in Pandas quickly run out of memory. Building scalable, reproducible, and fault-tolerant feature pipelines requires distributed computing with **Apache Spark** combined with disciplined **MLOps practices**.

Here is an architectural walkthrough of how to structure modular Spark pipelines, monitor feature drift using statistical tests, and seamlessly bridge batch feature stores with real-time inference microservices.

---

## 1. Modular Apache Spark Architecture

A common anti-pattern in large-scale data science is the "monolithic PySpark script" where extraction, cleaning, windowed aggregation, and model training are intertwined. 

To maintain production stability, our architecture decouples into distinct stages:

```
[ Raw Event Lake (HDFS / GCP Cloud Storage) ]
                      │
                      ▼
[ PySpark Data Extraction & Cleansing Job ]
                      │
                      ▼
[ Feature Transformation & Window Aggregations ]
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
[ Offline Feature Store ]    [ Online Cache (Redis) ]
(Parquet / Hive for Training) (Key-Value for Real-Time)
```

### PySpark Windowed Feature Generation Example

```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def generate_customer_temporal_features(df):
    """
    Computes rolling velocity and monetary aggregates per customer.
    Input df contains: customer_id, timestamp, transaction_amount
    """
    # Define time-based windows using unix epoch timestamps
    window_24h = (
        Window.partitionBy("customer_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-86400, 0)
    )
    
    window_30d = (
        Window.partitionBy("customer_id")
        .orderBy(F.col("timestamp").cast("long"))
        .rangeBetween(-86400 * 30, 0)
    )

    features_df = df.withColumn(
        "tx_count_24h", F.count("transaction_amount").over(window_24h)
    ).withColumn(
        "tx_sum_24h", F.sum("transaction_amount").over(window_24h)
    ).withColumn(
        "tx_avg_30d", F.avg("transaction_amount").over(window_30d)
    ).withColumn(
        "tx_ratio_current_to_30d_avg",
        F.col("transaction_amount") / (F.col("tx_avg_30d") + 1e-5)
    )

    return features_df
```

---

## 2. Statistical Data Drift Monitoring: PSI & KS-Tests

Even the best machine learning models degrade over time as consumer behavior, macroeconomic conditions, or system upstream schemas evolve. In regulated environments like banking and healthcare, silent model degradation cannot be tolerated.

### Population Stability Index (PSI)
We monitor distribution shift between baseline training features and incoming production data batches using the **Population Stability Index (PSI)**:

$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \cdot \ln\left( \frac{P_b}{Q_b} \right)$$

* **$\text{PSI} < 0.10$**: Stable distribution; no action needed.
* **$0.10 \le \text{PSI} < 0.25$**: Moderate shift; alert engineering team to evaluate.
* **$\text{PSI} \ge 0.25$**: Significant distributional shift; triggers automated fallback and retraining.

```python
import numpy as np

def calculate_psi(expected, actual, num_buckets=10):
    """
    Calculates the Population Stability Index between reference (expected)
    and production inference (actual) data arrays.
    """
    # Generate quantile bin edges from reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Bucket counts
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Fractions with smoothing epsilon
    p = (expected_counts / len(expected)) + 1e-4
    q = (actual_counts / len(actual)) + 1e-4

    # Normalize back to 1
    p /= np.sum(p)
    q /= np.sum(q)

    # PSI summation
    psi_value = np.sum((p - q) * np.log(p / q))
    return float(psi_value)
```

---

## 3. Microservices Deployment & Automated Governance

Once models are validated through CI/CD pipelines:
1. **Containerized Serialization:** Models are exported in optimized formats (ONNX, Treelite, or optimized scikit-learn pickles) packaged inside lightweight Docker containers running FastAPI or Triton Inference Server.
2. **Health Probes & Metrics:** Every container exports Prometheus telemetry tracking latency percentiles (p50, p95, p99), inference counts, and prediction distribution histograms.
3. **Model Governance Audit Logs:** In compliance with regulatory requirements, each model prediction logs feature inputs, model version hash, threshold applied, and response timestamp into immutable audit tables.

By combining distributed Spark batch computation with disciplined drift monitoring and microservices packaging, organizations achieve robust ML systems capable of processing millions of daily transactions without operational overhead.
