---
title: "Dense vs. Sparse Matrices: How Data Layout Decides Memory, Speed and Cloud Cost"
date: 2025-12-10T10:00:00+01:00
excerpt: "Why a model that fits on a laptop can crash a 64 GB server, and how choosing the right matrix format turns terabytes into gigabytes. The math of storage formats, a modular Python toolkit, and a worked text-search example."
categories:
  - Machine Learning
  - Data Engineering
tags:
  - Sparse Matrix
  - SciPy
  - Memory Optimization
  - Feature Engineering
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "table"
---

Some of the most expensive problems I have fixed in production had nothing to do with the model. The
model was fine. The data was simply stored in the wrong shape. A feature matrix that should have
needed two gigabytes was being held as eight terabytes of mostly zeros, and the team was about to
buy bigger machines to solve it. This post explains how to recognise that situation, the math behind
the fix, and a small Python toolkit I use to make the decision explicitly instead of by accident.

## The business problem

Whenever data describes *which things relate to which other things*, most of the possible
combinations never happen:

- Customers and the products they bought: a shop with 500,000 products, where a typical customer
  has bought 30 of them.
- Documents and the words they contain: a vocabulary of 300,000 terms, where a typical document
  uses 200.
- One-hot encoded categories: a "postcode" column with 10,000 values becomes 10,000 columns, and each
  row has exactly one 1.

If you store these as ordinary (dense) tables, you pay for every zero, in RAM, in disk, in network
transfer and in CPU time. In the cloud, that turns directly into a monthly bill and into jobs that
fail with "out of memory" at 2 a.m. Choosing the right format is one of the cheapest performance
wins available to a data team.

## The idea in plain English

A **dense** matrix stores every cell, whether it holds a useful number or a zero. It is like a
printed spreadsheet: every box exists on paper even if it is empty.

A **sparse** matrix stores only the cells that are not zero, together with their coordinates. It is
like a list that says "row 7, column 412: value 3". Everything not on the list is assumed to be zero.

When most cells are zero, the list is far shorter than the spreadsheet. When most cells are filled,
the list becomes longer than the spreadsheet, because you are paying for coordinates as well as
values. So the right choice depends on one number: how full the matrix is.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Raw data<small>interactions, text, categories</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">Measure density<small>non-zeros / all cells</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-branch">
      <div class="dg-node dg-ok">Sparse path<small>build COO, convert once to CSR</small></div>
      <div class="dg-node">Dense path<small>plain NumPy array</small></div>
    </div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Sparse-safe pipeline<small>guards after every step</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Model or search index</div>
  </div>
  <p class="dg-note">Below about 1% density go sparse, above about 10% go dense, in between measure both.</p>
  <p class="dg-caption">Choosing the storage format from the data, not by habit.</p>
</div>

## How it works: the math, step by step

### Density

Let the matrix have $$m$$ rows, $$n$$ columns and $$\text{nnz}$$ non-zero entries. Its density is

$$
d = \frac{\text{nnz}}{m \cdot n}
$$

In plain English: the fraction of cells that actually hold information. A density of 0.001 means one
cell in a thousand is non-zero.

### Memory of a dense matrix

If every value takes $$b$$ bytes (8 bytes for a 64-bit float), the dense matrix needs

$$
M_{\text{dense}} = m \cdot n \cdot b
$$

Read it as: every row times every column times the size of one number. Zeros cost exactly as much as
any other value.

### Memory of a CSR matrix

The most common sparse format is **Compressed Sparse Row (CSR)**. It keeps three arrays:

1. `data`: the non-zero values, length $$\text{nnz}$$.
2. `indices`: the column of each value, length $$\text{nnz}$$.
3. `indptr`: where each row starts inside `data`, length $$m + 1$$.

With $$i$$ bytes per integer index (4 bytes for int32), the memory is

$$
M_{\text{CSR}} = \text{nnz} \cdot (b + i) + (m + 1) \cdot i
$$

Line by line: each stored value costs its own bytes plus the bytes for its column number, and each
row costs one extra integer that marks where it begins.

### The break-even density

CSR is smaller than dense when $$M_{\text{CSR}} < M_{\text{dense}}$$. If we ignore the small `indptr`
term and divide both sides by $$m \cdot n$$:

$$
d \cdot (b + i) < b
\quad\Longrightarrow\quad
d < \frac{b}{b + i}
$$

For float64 values and int32 indices this gives $$d < 8/12 \approx 0.67$$. In plain English: on
memory alone, sparse storage wins until the matrix is about two-thirds full.

In practice the useful threshold for **speed** is much lower, roughly 5 to 10 percent, because sparse
arithmetic jumps around in memory and modern CPUs are very fast at the regular, predictable access
pattern of dense arrays. My rule of thumb: below 1 percent density, always go sparse; above 10
percent, usually go dense; in between, measure.

### Cost of the core operation

Most models spend their time multiplying a matrix by a vector, $$y = A x$$. For a dense matrix every
cell is touched:

$$
\text{cost}_{\text{dense}} = O(m \cdot n)
$$

For CSR, only the stored values are touched:

$$
\text{cost}_{\text{CSR}} = O(\text{nnz}) = O(d \cdot m \cdot n)
$$

So at a density of 0.0001 the sparse multiplication does ten thousand times less work. This is why
linear models, logistic regression and many recommender algorithms train comfortably on sparse
inputs that would be impossible to hold densely.

### Which sparse format for which job

| Format | Good at | Weak at |
|---|---|---|
| COO (coordinate list) | Building a matrix from (row, col, value) triples | Arithmetic, slicing |
| CSR (compressed rows) | Row slicing, matrix times vector, most scikit-learn models | Changing the structure, column slicing |
| CSC (compressed columns) | Column slicing, some solvers | Row slicing |
| LIL / DOK | Inserting values one by one | Everything else |

The usual workflow is: build in COO, convert once to CSR, then compute.

## Implementation in Python

I keep the decision logic in one small, testable module rather than scattering `toarray()` calls
around notebooks.

```python
# sparse_toolkit.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class MatrixProfile:
    rows: int
    cols: int
    nnz: int
    value_bytes: int = 8   # float64
    index_bytes: int = 4   # int32

    @property
    def density(self) -> float:
        return self.nnz / (self.rows * self.cols)

    @property
    def dense_bytes(self) -> int:
        return self.rows * self.cols * self.value_bytes

    @property
    def csr_bytes(self) -> int:
        return self.nnz * (self.value_bytes + self.index_bytes) + (self.rows + 1) * self.index_bytes

    def recommend(self, speed_threshold: float = 0.05) -> str:
        """Return 'sparse' or 'dense' based on density and memory."""
        if self.csr_bytes >= self.dense_bytes:
            return "dense"
        return "sparse" if self.density < speed_threshold else "dense"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


def profile(matrix: np.ndarray | sp.spmatrix) -> MatrixProfile:
    rows, cols = matrix.shape
    nnz = matrix.nnz if sp.issparse(matrix) else int(np.count_nonzero(matrix))
    return MatrixProfile(rows=rows, cols=cols, nnz=nnz)


def from_triples(rows: np.ndarray, cols: np.ndarray, values: np.ndarray,
                 shape: tuple[int, int]) -> sp.csr_matrix:
    """Build in COO (cheap to construct), return CSR (cheap to compute with)."""
    coo = sp.coo_matrix((values, (rows, cols)), shape=shape)
    csr = coo.tocsr()
    csr.sum_duplicates()
    return csr
```

Using it on a realistic interaction matrix:

```python
import numpy as np
from sparse_toolkit import MatrixProfile, from_triples, human_bytes

rng = np.random.default_rng(42)
n_customers, n_products, events = 200_000, 50_000, 3_000_000

rows = rng.integers(0, n_customers, size=events)
cols = rng.integers(0, n_products, size=events)
qty = np.ones(events, dtype=np.float64)

X = from_triples(rows, cols, qty, shape=(n_customers, n_products))
p = MatrixProfile(*X.shape, nnz=X.nnz)

print(f"density      {p.density:.6f}")
print(f"dense        {human_bytes(p.dense_bytes)}")
print(f"csr          {human_bytes(p.csr_bytes)}")
print(f"recommended  {p.recommend()}")
```

On this example the dense version would need about 75 GB, while the CSR version needs about 35 MB.

### Keeping data sparse through a scikit-learn pipeline

Most of the damage happens when one step quietly converts the matrix to dense. These choices keep it
sparse end to end:

```python
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(text_col: str, cat_cols: list[str], num_cols: list[str]) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(min_df=5, max_features=200_000), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            # with_mean=False: centering would fill every zero and destroy sparsity
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        sparse_threshold=1.0,  # always return a sparse matrix
    )
    model = LogisticRegression(solver="saga", max_iter=2000)
    return Pipeline([("features", features), ("model", model)])
```

Three details matter here. `sparse_output=True` keeps one-hot columns sparse. `with_mean=False`
stops the scaler from subtracting the mean, which would turn every zero into a non-zero.
`sparse_threshold=1.0` tells the column transformer never to fall back to a dense result.

### A guard I add to production code

```python
import scipy.sparse as sp


def assert_sparse(X, name: str, max_density: float = 0.05) -> None:
    if not sp.issparse(X):
        raise TypeError(f"{name} became dense; check the previous transformation step")
    density = X.nnz / (X.shape[0] * X.shape[1])
    if density > max_density:
        raise ValueError(f"{name} density {density:.3f} is higher than expected")
```

This one function has saved me more debugging time than any profiler. It fails loudly at the step
that broke sparsity, instead of failing later with an out-of-memory error far from the cause.

## A worked example: product search over two million documents

Say a company wants keyword search and "similar items" over 2 million product descriptions. After
cleaning, the vocabulary has 500,000 terms and the average description uses 100 distinct terms.

- Dense TF-IDF matrix: $$2{,}000{,}000 \times 500{,}000 \times 8$$ bytes, which is 8 TB. No single
  machine holds that.
- Non-zeros: $$2{,}000{,}000 \times 100 = 2 \times 10^8$$.
- CSR matrix: $$2 \times 10^8 \times 12 + 2{,}000{,}001 \times 4$$ bytes, which is about 2.4 GB.

The same information goes from "impossible" to "fits on a laptop". Similarity search becomes a
sparse matrix product, and the whole index rebuilds in minutes on a single mid-size cloud instance
rather than requiring a cluster. That is a direct reduction in infrastructure cost, and in how long
the team waits for each experiment.

## Pitfalls and when not to use sparse

- **Calling `.toarray()` or `.todense()` "just to look".** On a large matrix this is the out-of-memory
  error. Inspect `X[:5]` instead.
- **Centering or standardising with the mean.** Subtracting a mean fills every zero. Use scalers that
  do not center, or center inside the model.
- **Column slicing on CSR.** It is slow. Convert to CSC once if you need many column operations.
- **Changing the structure of CSR in a loop.** Each insertion rebuilds arrays. Collect triples and
  build once.
- **Algorithms that do not support sparse input.** Many tree-based libraries and most deep learning
  layers want dense batches. For those, either densify small batches on the fly or reduce
  dimensionality first (for example with truncated SVD).
- **Matrices that are not actually sparse.** Images, sensor readings and dense embeddings are mostly
  non-zero. Sparse storage there costs more memory and runs slower.

## Measuring success in production

- Peak memory of the training and scoring jobs, before and after.
- Job runtime and the instance size needed to run it.
- Monthly compute cost for the pipeline.
- Number of out-of-memory failures per month (the target is zero).

## Checklist

- Compute density for every large feature matrix and log it.
- Build from triples in COO, convert once to CSR.
- Use sparse-friendly encoders and scalers (`sparse_output=True`, `with_mean=False`).
- Add an `assert_sparse` guard after each transformation step.
- Choose CSR for row access and model training, CSC for column access.
- Measure speed between 1 and 10 percent density instead of guessing.

## Final thought

Storage format is a design decision, not a detail. Getting it right often removes the need for a
bigger machine, a cluster, or a rewrite. If your team is fighting memory limits or slow pipelines,
I am happy to take a look. See [how I work with data and ML platforms](/services/#production-mlops)
or [get in touch](/contact/).
