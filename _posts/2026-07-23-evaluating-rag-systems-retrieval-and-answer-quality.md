---
title: "How to Evaluate a RAG System: Retrieval Metrics, Grounding Checks and a Test Set You Can Trust"
date: 2026-07-23T10:00:00+02:00
excerpt: "If you cannot measure whether your AI assistant is right, you cannot improve it or defend it. A practical evaluation framework for retrieval-augmented generation: the math behind recall@k, MRR and nDCG, how to check that answers are grounded in sources, and how to compare two versions with confidence."
categories:
  - Generative AI
  - RAG
tags:
  - RAG
  - Evaluation
  - Information Retrieval
  - LLM
  - Bootstrap
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "clipboard-check"
---

The question I am asked most often about AI assistants is "how good is it?", and the honest answer for
most systems I review is "nobody knows". Changes are made because a demo looked better, a new model
was released, or someone tried three questions and liked the answers. That is not engineering. This
post describes the evaluation framework I set up before tuning any retrieval-augmented generation
(RAG) system: a test set built from real questions, retrieval metrics with clear meaning, a grounding
check for the generated answers, and a statistical comparison that tells you whether a change really
helped.

## The business problem

A RAG assistant is usually approved on the strength of a demo. Once it is live, the organisation needs
answers to harder questions:

- Is it accurate enough to put in front of customers, or only staff?
- Did last week's change (new chunking, new model, new prompt) make it better or worse?
- Where does it fail: finding the right document, or answering from it?
- Can we show an auditor or a manager evidence that it behaves as intended?

Without measurement, every one of these becomes an opinion. Teams either freeze the system out of fear
of breaking it, or keep changing it without knowing whether they are moving forward. Both are
expensive. A small, well designed evaluation suite turns these debates into numbers, and it is also
the strongest argument for further investment when the numbers are good.

## The idea in plain English

A RAG system can fail in two separate places, so we measure them separately:

1. **Retrieval.** Did the system find the passages that contain the answer? If not, no language model
   can save it.
2. **Generation.** Given the passages it found, did it answer correctly, and only from those passages?

For retrieval we need, for each test question, a list of which passages are relevant. Then we check
where those passages appear in the ranked results. For generation we break each answer into small
factual claims and check whether each claim is supported by the retrieved text.

Finally, because test sets are finite, a difference of two percentage points might be luck. A simple
resampling method tells us how confident we can be that version B is really better than version A.

## How it works: the math, step by step

Let the test set have $$N$$ questions. For question $$i$$, let $$R_i$$ be the set of relevant
passages and let the system return a ranked list $$(d_{i,1}, d_{i,2}, \dots)$$.

### Recall at k

$$
\text{Recall@}k = \frac{1}{N} \sum_{i=1}^{N} \frac{\big\lvert\{d_{i,1}, \dots, d_{i,k}\} \cap R_i\big\rvert}{\lvert R_i \rvert}
$$

In plain English: for each question, what share of the relevant passages made it into the top
$$k$$, averaged over questions. If $$k$$ is the number of passages you send to the language model,
this is the most important retrieval number, because anything not in the top $$k$$ is invisible to
the answer.

### Precision at k

$$
\text{Precision@}k = \frac{1}{N} \sum_{i=1}^{N} \frac{\big\lvert\{d_{i,1}, \dots, d_{i,k}\} \cap R_i\big\rvert}{k}
$$

In plain English: what share of what we send to the model is actually relevant. Low precision means
the model must ignore noise, which increases cost and the chance of a confused answer.

### Mean Reciprocal Rank

Let $$\text{rank}_i$$ be the position of the first relevant passage for question $$i$$. Then

$$
\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}
$$

Read it as: a first relevant result at position 1 scores 1, at position 2 scores 0.5, at position 5
scores 0.2. If nothing relevant is found, the score is 0. MRR rewards putting the best passage at the
very top.

### Normalised Discounted Cumulative Gain

When relevance is graded (for example 2 = directly answers, 1 = useful context, 0 = irrelevant), we
use nDCG. With $$g_{i,j}$$ the grade of the passage at position $$j$$:

$$
\text{DCG@}k_i = \sum_{j=1}^{k} \frac{2^{g_{i,j}} - 1}{\log_2(j + 1)}
\qquad
\text{nDCG@}k_i = \frac{\text{DCG@}k_i}{\text{IDCG@}k_i}
$$

Line by line:

- $$2^{g} - 1$$ turns grades into gains, so a highly relevant passage is worth much more than a
  partly relevant one.
- Dividing by $$\log_2(j + 1)$$ discounts lower positions, because readers and models pay less
  attention to them.
- $$\text{IDCG}$$ is the best possible DCG for that question (the same passages in the ideal order),
  so nDCG is between 0 and 1 and can be averaged across questions.

### Faithfulness of the answer

Split each generated answer into atomic claims $$c_1, \dots, c_m$$ (short, single facts). For each
claim, decide whether it is supported by the retrieved context. Then

$$
\text{Faithfulness}_i = \frac{\text{number of supported claims}}{m}
$$

In plain English: what share of what the assistant said can actually be found in its sources. A
faithfulness of 0.8 means one claim in five came from somewhere else, usually the model's own
training data or imagination. For regulated or customer-facing uses, I look at the share of answers
with faithfulness equal to 1, not just the average.

The support decision can be made by people on a sample, or by a second language model acting as a
judge. When a model is used as a judge, I always measure its agreement with human labels on a subset
first, for example with Cohen's kappa

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where $$p_o$$ is the observed agreement and $$p_e$$ the agreement expected by chance. A kappa above
about 0.7 is good enough to rely on the judge for day-to-day comparisons.

### Is version B really better? The paired bootstrap

Let $$s_i^{A}$$ and $$s_i^{B}$$ be a per-question metric (for example reciprocal rank) for the two
versions, and $$\delta_i = s_i^{B} - s_i^{A}$$. The observed improvement is
$$\bar{\delta} = \frac{1}{N}\sum_i \delta_i$$. To estimate its uncertainty:

1. Draw $$N$$ questions with replacement from the test set.
2. Compute the mean of $$\delta$$ on that resample.
3. Repeat $$B$$ times (for example 10,000).

The 2.5th and 97.5th percentiles of the resampled means form a 95 percent confidence interval for the
improvement. In plain English: if the whole interval is above zero, the improvement is very unlikely
to be luck from the particular questions we happened to pick. Pairing (comparing both versions on the
same questions) removes the noise of question difficulty, which makes the test much more sensitive.

## Implementation in Python

### The test set

```python
# rag_eval/dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalItem:
    qid: str
    question: str
    relevant: dict[str, int]                 # passage_id -> grade (1 or 2)
    reference_answer: str = ""
    tags: list[str] = field(default_factory=list)  # e.g. "exact-code", "multi-hop", "out-of-scope"


def load_items(path: str | Path) -> list[EvalItem]:
    with open(path, encoding="utf-8") as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]
```

I build the first version of this file from real questions: search logs, service desk tickets, or a
workshop with the people who will use the system. I deliberately include out-of-scope questions with
an empty `relevant` set, because declining correctly is part of the job.

### Retrieval metrics

```python
# rag_eval/retrieval_metrics.py
from __future__ import annotations

import math


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def precision_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked[:k]) & set(relevant)) / k


def reciprocal_rank(ranked: list[str], relevant: dict[str, int]) -> float | None:
    if not relevant:
        return None
    for pos, pid in enumerate(ranked, start=1):
        if pid in relevant:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float | None:
    if not relevant:
        return None
    dcg = sum((2 ** relevant.get(pid, 0) - 1) / math.log2(j + 1)
              for j, pid in enumerate(ranked[:k], start=1))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(j + 1) for j, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
```

Returning `None` for out-of-scope questions keeps them out of the retrieval averages; they are scored
separately on whether the system declined.

### Grounding check with a judge

```python
# rag_eval/faithfulness.py
from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, ValidationError


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool


class Judgement(BaseModel):
    claims: list[ClaimVerdict]


JUDGE_PROMPT = (
    "Split the ANSWER into short factual claims. For each claim, decide if it is fully supported by "
    "the CONTEXT. Respond with JSON: "
    '{"claims": [{"claim": "...", "supported": true}]}. Do not use outside knowledge.'
)


def faithfulness(llm: LLM, answer: str, context: str) -> float | None:
    raw = llm.complete(JUDGE_PROMPT, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}")
    try:
        judgement = Judgement.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError):
        return None  # count separately as a judge failure
    if not judgement.claims:
        return 1.0
    return sum(c.supported for c in judgement.claims) / len(judgement.claims)
```

### Running an evaluation

```python
# rag_eval/run.py
from __future__ import annotations

from statistics import mean
from typing import Callable

import pandas as pd

from rag_eval.dataset import EvalItem
from rag_eval.retrieval_metrics import ndcg_at_k, recall_at_k, reciprocal_rank

SearchFn = Callable[[str], list[str]]  # question -> ranked passage ids


def evaluate_retrieval(search: SearchFn, items: list[EvalItem], k: int = 5) -> pd.DataFrame:
    rows = []
    for it in items:
        ranked = search(it.question)
        rows.append({
            "qid": it.qid,
            "tags": ",".join(it.tags),
            f"recall@{k}": recall_at_k(ranked, it.relevant, k),
            "rr": reciprocal_rank(ranked, it.relevant),
            f"ndcg@{k}": ndcg_at_k(ranked, it.relevant, k),
        })
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict[str, float]:
    metric_cols = [c for c in df.columns if c not in ("qid", "tags")]
    return {c: round(mean(df[c].dropna()), 3) for c in metric_cols}
```

### Comparing two versions

```python
# rag_eval/compare.py
import numpy as np
import pandas as pd


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 10_000, seed: int = 0) -> dict[str, float]:
    """a, b: per-question metric for versions A and B, indexed by qid."""
    joined = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    delta = (joined["b"] - joined["a"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot_means = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"mean_delta": float(delta.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p_b_better": float((boot_means > 0).mean())}
```

Usage reads naturally:

```python
base = evaluate_retrieval(search_v1, items).set_index("qid")
new = evaluate_retrieval(search_v2, items).set_index("qid")
print(summary(base.reset_index()), summary(new.reset_index()))
print(paired_bootstrap(base["rr"], new["rr"]))
```

## A worked example: choosing a chunking strategy

A team running an internal assistant over 3,000 technical manuals wants to switch from fixed
500-token chunks to heading-based chunks. They build a test set of 240 real questions from the service
desk, including 30 that the manuals do not answer.

Results on the 210 answerable questions:

| Metric | Fixed chunks | Heading chunks |
|---|---|---|
| Recall@5 | 0.71 | 0.80 |
| MRR | 0.58 | 0.66 |
| nDCG@5 | 0.62 | 0.70 |

The paired bootstrap on reciprocal rank gives a mean improvement of 0.08 with a 95 percent interval
of roughly 0.04 to 0.12. The whole interval is above zero, so the team can be confident the change is
real. Breaking the results down by tag shows where it came from: questions tagged `table` improved the
most, because heading-based chunks no longer cut tables in half.

Faithfulness, measured with a judge that agreed with human reviewers at $$\kappa = 0.78$$ on a sample
of 60 answers, rose from 0.88 to 0.93. On the 30 out-of-scope questions, the new version correctly
declined 27 times instead of 22.

That is a decision backed by evidence, and a report the team can show to the system owner.

## Pitfalls and when not to use it

- **Synthetic questions only.** Questions generated by a model from the documents are easy and look
  like the documents. Mix in real user questions, or your scores will be too optimistic.
- **Labelling only one relevant passage.** Many questions are answered in several places. Missing
  labels make recall look worse than it is. Review the top results of a strong system and add any
  missed relevant passages.
- **A judge you never checked.** Always measure judge agreement with humans before trusting it.
- **Optimising the average only.** Report by tag (exact codes, tables, multi-step questions,
  out-of-scope). Averages hide the failure modes that cause complaints.
- **Test set leakage.** If you tune prompts on the same questions you report, you are overfitting.
  Keep a frozen hold-out set.
- **Small test sets with big claims.** With 30 questions, confidence intervals are wide. Aim for at
  least 150 to 200 for decisions that matter.

## Measuring success in production

Offline evaluation is the gate; production monitoring is the check:

- Share of answers with citations, and share of questions declined.
- User feedback rate and the share of negative feedback, broken down by topic.
- A weekly sample of live answers scored for faithfulness.
- New question types appearing in logs that the test set does not cover yet (add them).

## Checklist

- Build a test set from real questions, including out-of-scope ones.
- Label relevant passages, with grades if possible.
- Track recall@k, MRR and nDCG for every retrieval change.
- Score faithfulness with a judge that has been validated against people.
- Compare versions with a paired bootstrap, not with a gut feeling.
- Report results by question type, not only as one average.
- Keep a frozen hold-out set for final decisions.

## Final thought

Evaluation is what turns an AI assistant from a demo into a product. It is also what makes
conversations with stakeholders easier: instead of "it seems better", you can say "recall improved
from 71 to 80 percent, and we are 95 percent confident the gain is real". If you need an evaluation
framework for an assistant you have built or are planning, see my
[generative AI and RAG service](/services/#enterprise-generative-ai) or [contact me](/contact/).
