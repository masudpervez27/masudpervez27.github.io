---
title: "Production RAG: Hybrid Retrieval, Reranking and a Self-Correcting Loop with LangGraph"
date: 2026-03-18T10:00:00+01:00
excerpt: "Why the tutorial version of retrieval-augmented generation breaks on real company documents, and how hybrid search, reranking and a small self-correcting graph fix it. The math of each ranking step, a modular Python build, and an internal policy assistant example."
categories:
  - Generative AI
  - RAG
tags:
  - RAG
  - LangGraph
  - Hybrid Search
  - BM25
  - Embeddings
  - Reranking
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "brain"
---

The first retrieval-augmented generation (RAG) demo always looks impressive. You load a few PDFs,
ask a question, and get a fluent answer. Then real users arrive with real questions: an exact product
code, a clause number, a policy that was updated last month. The answers start to sound confident and
be wrong. In this post I walk through the retrieval design I use to close that gap: hybrid search,
reciprocal rank fusion, a cross-encoder reranker, and a small LangGraph loop that checks its own
evidence before answering.

## The business problem

Most organisations have their knowledge spread across policies, manuals, contracts, tickets and wiki
pages. Staff spend a surprising part of their day searching for answers, or asking the one colleague
who knows. A RAG assistant can answer those questions in seconds, *if* it finds the right passage.

When retrieval is weak, three things happen:

- **Wrong answers that sound right.** The language model fills the gap with plausible text. In
  customer-facing or compliance settings, that is a real risk.
- **No trust, no adoption.** After two or three bad answers, people go back to asking colleagues and
  the investment is wasted.
- **Rising cost.** Teams compensate by stuffing more documents into the prompt, which increases token
  costs and latency without fixing the cause.

The fix is almost always in retrieval, not in the language model.

## The idea in plain English

A RAG system answers a question in two steps: first it *finds* the most relevant passages, then it
asks a language model to *answer using only those passages*. The quality of the answer is capped by
the quality of what was found.

Good retrieval combines two kinds of search:

- **Keyword search** is excellent at exact matches: article numbers, error codes, names, legal
  references.
- **Semantic search** with embeddings is excellent at meaning: "how do I get reimbursed for travel"
  should find a passage titled "expense claims".

Neither is enough on its own, so we run both, merge their result lists, and then let a more careful
model re-read the top candidates and put them in the best order. Finally, before answering, the
system checks whether the evidence actually answers the question. If not, it rewrites the query and
tries again, or says honestly that it does not know.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Question</div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-branch">
      <div class="dg-node">Keyword search<small>BM25</small></div>
      <div class="dg-node">Vector search<small>embeddings</small></div>
    </div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Rank fusion<small>RRF</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">Reranker<small>cross-encoder on top 30</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Score check<small>minimum relevance</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-branch">
      <div class="dg-node dg-ok">Answer with citations</div>
      <div class="dg-node dg-warn">Decline honestly</div>
    </div>
  </div>
  <p class="dg-note">If nothing passes the score check, the query is rewritten and retrieval runs again, at most twice.</p>
  <p class="dg-caption">Hybrid retrieval with reranking and a self-correcting loop.</p>
</div>

## How it works: the math, step by step

### Keyword search with BM25

For a query $$q$$ with terms $$t$$, the BM25 score of a document $$d$$ is

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot
\frac{f(t, d)\,(k_1 + 1)}{f(t, d) + k_1 \left(1 - b + b \frac{\lvert d \rvert}{\text{avgdl}}\right)}
$$

with

$$
\text{IDF}(t) = \ln\!\left(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\right)
$$

Line by line:

- $$f(t, d)$$ is how many times the term appears in the document. More mentions mean more relevance,
  but the fraction saturates, so the tenth mention adds much less than the first. $$k_1$$ (often 1.2
  to 2.0) controls how fast it saturates.
- $$\lvert d \rvert / \text{avgdl}$$ compares the document length with the average. Long documents mention many
  words by chance, so $$b$$ (often 0.75) penalises them.
- $$\text{IDF}(t)$$ rewards rare terms. $$N$$ is the number of documents and $$n_t$$ the number
  containing the term. A word like "the" appears everywhere and scores almost nothing; a product code
  appears in two documents and scores highly.

### Semantic search with embeddings

An embedding model maps each text to a vector. Relevance is the cosine similarity between the query
vector $$\mathbf{q}$$ and the passage vector $$\mathbf{d}$$:

$$
\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{d} \rVert}
$$

In plain English: two texts are similar when their vectors point in the same direction, regardless
of their length. If we normalise all vectors to length one, this is just a dot product, which vector
databases compute very fast.

### Merging the lists with Reciprocal Rank Fusion

BM25 scores and cosine similarities live on different scales, so adding them is meaningless. Instead
we combine **ranks**. If document $$d$$ is at position $$r_i(d)$$ in result list $$i$$, its fused
score is

$$
\text{RRF}(d) = \sum_{i} \frac{1}{k + r_i(d)}
$$

with $$k = 60$$ as a common default.

Read it as: each list gives the document a vote, and a higher position gives a bigger vote. The
constant $$k$$ flattens the curve so that being first instead of second matters, but not so much that
one list can dominate. A document that is fairly high in *both* lists beats one that is first in one
list and absent from the other. That is exactly the behaviour we want.

### Reranking with a cross-encoder

The first two retrievers encode the query and the passage separately, which is fast but shallow. A
**cross-encoder** reads the query and passage together and outputs a relevance score:

$$
s(q, d) = \sigma\big(\mathbf{w}^{\top} h_{\text{[CLS]}}(q \oplus d)\big)
$$

In plain English: glue the question and passage into one input, let a transformer read them
together, take its summary vector $$h_{\text{[CLS]}}$$, and turn it into a score between 0 and 1 with
the sigmoid $$\sigma$$. Because it sees both texts at once, it notices details such as negation
("not eligible") that vector similarity misses. It is too slow to run on the whole collection, so we
only run it on the top 20 to 50 fused candidates.

### Chunking

Documents are split into chunks of about $$L$$ tokens with an overlap of $$o$$ tokens. The number of
chunks for a document of $$T$$ tokens is

$$
n_{\text{chunks}} = \left\lceil \frac{T - o}{L - o} \right\rceil
$$

In plain English: each new chunk moves forward by $$L - o$$ tokens, so the overlap ensures that a
sentence cut at a boundary still appears whole in one chunk. I prefer splitting on headings and
paragraphs first and only then by length, and I store the section title with each chunk.

## Implementation in Python

The code is split into small parts with clear interfaces, so each part can be swapped (another vector
store, another reranker, another LLM provider) without touching the rest.

### Data model and chunking

```python
# rag/chunking.py
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


def split_by_headings(doc_id: str, markdown: str, max_words: int = 220,
                      overlap_words: int = 40) -> list[Chunk]:
    """Split on headings first, then by length with overlap."""
    sections = re.split(r"\n(?=#{1,4} )", markdown)
    chunks: list[Chunk] = []
    for s_idx, section in enumerate(sections):
        lines = section.strip().splitlines()
        if not lines:
            continue
        title = lines[0].lstrip("# ").strip()
        words = " ".join(lines[1:]).split()
        if not words:
            continue
        step = max_words - overlap_words
        for start in range(0, max(len(words) - overlap_words, 1), step):
            text = " ".join(words[start:start + max_words])
            chunks.append(Chunk(f"{doc_id}:{s_idx}:{start}", doc_id, title, text))
    return chunks
```

### Retrievers

```python
# rag/retrievers.py
from __future__ import annotations

import re
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.chunking import Chunk


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[str]: ...


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-_.]*", text.lower())


class KeywordRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(f"{c.title} {c.text}") for c in chunks])

    def search(self, query: str, k: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [self.ids[i] for i in top if scores[i] > 0]


class VectorRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = "BAAI/bge-small-en-v1.5"):
        self.ids = [c.chunk_id for c in chunks]
        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode(
            [f"{c.title}\n{c.text}" for c in chunks], normalize_embeddings=True
        )

    def search(self, query: str, k: int) -> list[str]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [self.ids[i] for i in top]
```

In production the `VectorRetriever` would sit on a vector database (pgvector, Qdrant, OpenSearch and
so on). Because it implements the same `search` method, nothing else changes.

### Fusion and reranking

```python
# rag/ranking.py
from __future__ import annotations

from collections import defaultdict

from sentence_transformers import CrossEncoder


def reciprocal_rank_fusion(result_lists: list[list[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for results in result_lists:
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_n: int) -> list[tuple[str, float]]:
        """candidates: (chunk_id, text). Returns (chunk_id, score) sorted by score."""
        if not candidates:
            return []
        scores = self.model.predict([(query, text) for _, text in candidates])
        ranked = sorted(zip((cid for cid, _ in candidates), scores), key=lambda x: -x[1])
        return [(cid, float(s)) for cid, s in ranked[:top_n]]
```

### The retrieval pipeline

```python
# rag/pipeline.py
from __future__ import annotations

from dataclasses import dataclass

from rag.chunking import Chunk
from rag.ranking import Reranker, reciprocal_rank_fusion
from rag.retrievers import KeywordRetriever, VectorRetriever


@dataclass
class RetrievalConfig:
    k_each: int = 30
    k_rerank: int = 30
    top_n: int = 5
    min_score: float = 0.0  # in the reranker's own score units; tune it on a labelled test set


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], cfg: RetrievalConfig = RetrievalConfig()):
        self.by_id = {c.chunk_id: c for c in chunks}
        self.keyword = KeywordRetriever(chunks)
        self.vector = VectorRetriever(chunks)
        self.reranker = Reranker()
        self.cfg = cfg

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        fused = reciprocal_rank_fusion([
            self.keyword.search(query, self.cfg.k_each),
            self.vector.search(query, self.cfg.k_each),
        ])[: self.cfg.k_rerank]
        candidates = [(cid, f"{self.by_id[cid].title}\n{self.by_id[cid].text}") for cid in fused]
        ranked = self.reranker.rerank(query, candidates, self.cfg.top_n)
        return [(self.by_id[cid], s) for cid, s in ranked if s >= self.cfg.min_score]
```

Note that `min_score` is applied after reranking. That is what allows the system to say "I could not
find this" instead of passing weak passages to the language model.

### A self-correcting loop with LangGraph

The last piece decides what to do with the evidence. I model it as a small graph: retrieve, grade,
and then either answer, rewrite the query and try again, or decline.

```python
# rag/graph.py
from __future__ import annotations

from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from rag.pipeline import HybridRetriever


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class RAGState(TypedDict):
    question: str
    query: str
    passages: list[dict]
    attempts: int
    answer: str


ANSWER_PROMPT = (
    "Answer the question using only the numbered sources. "
    "Cite sources like [1]. If the sources do not contain the answer, say so."
)
REWRITE_PROMPT = "Rewrite the search query to find the missing information. Return only the query."


def build_graph(retriever: HybridRetriever, llm: LLM, max_attempts: int = 2):
    def retrieve(state: RAGState) -> dict:
        hits = retriever.retrieve(state["query"])
        passages = [{"id": c.chunk_id, "title": c.title, "text": c.text, "score": s} for c, s in hits]
        return {"passages": passages, "attempts": state["attempts"] + 1}

    def route(state: RAGState) -> str:
        if state["passages"]:
            return "answer"
        return "rewrite" if state["attempts"] < max_attempts else "decline"

    def rewrite(state: RAGState) -> dict:
        user = f"Question: {state['question']}\nPrevious query: {state['query']}"
        return {"query": llm.complete(REWRITE_PROMPT, user).strip()}

    def answer(state: RAGState) -> dict:
        sources = "\n\n".join(
            f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(state["passages"], start=1)
        )
        user = f"Sources:\n{sources}\n\nQuestion: {state['question']}"
        return {"answer": llm.complete(ANSWER_PROMPT, user)}

    def decline(state: RAGState) -> dict:
        return {"answer": "I could not find this in the available documents. "
                          "Please contact the document owner or rephrase the question."}

    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("rewrite", rewrite)
    g.add_node("answer", answer)
    g.add_node("decline", decline)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route, {"answer": "answer", "rewrite": "rewrite", "decline": "decline"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", END)
    g.add_edge("decline", END)
    return g.compile()


def ask(app, question: str) -> str:
    state = app.invoke({"question": question, "query": question, "passages": [], "attempts": 0, "answer": ""})
    return state["answer"]
```

The `LLM` protocol keeps the graph independent of any vendor. An adapter for your provider of choice
is a few lines that implement `complete(system, user)`.

## A worked example: an internal policy assistant

A company with 900 employees has around 400 policy and process documents: travel, procurement, IT
security, HR. The HR and IT service desks receive about 1,500 questions a month, many of them
repeated.

- **Ingestion.** Documents are converted to markdown, split by headings into about 6,000 chunks, and
  each chunk keeps its document title, section, owner and last-updated date as metadata.
- **Retrieval.** A question like "Can I book business class for a 7-hour flight?" matches the travel
  policy semantically, while "What is the approval limit in PO-114?" is found by BM25 because of the
  exact code. RRF merges both, and the cross-encoder puts the passage with the actual rule at the top.
- **Answering.** The answer cites the section, so the employee can click through and check it.
- **Declining.** For "What is the parental leave policy in our Oslo office?", nothing scores above
  the threshold because the Norwegian policy was never uploaded. The assistant says so and points to
  HR, instead of inventing an answer from the Swedish policy.

A realistic outcome for a system like this is that a large share of repetitive questions are answered
without a ticket, and the service desk sees fewer, more complex cases. Exact numbers depend on the
quality and coverage of the documents, which is why I always start by measuring retrieval quality on
a test set (the subject of a [separate post on RAG evaluation](/posts/2026-07-23-evaluating-rag-systems-retrieval-and-answer-quality/)).

## Pitfalls and when not to use it

- **Bad source documents.** RAG cannot fix outdated or contradictory documents. Often the first win
  is cleaning up the document set and assigning owners.
- **No permission model.** If every user can retrieve every chunk, the assistant can leak
  confidential content. Filter by the user's access rights *at retrieval time*, not in the prompt.
- **Tables and scanned PDFs.** Plain text extraction destroys table structure. Use layout-aware
  parsing and store tables as markdown.
- **Skipping evaluation.** Without a test set of real questions with known answers, every change is a
  guess.
- **When a database query is the right tool.** "How many orders did we ship last week?" is a SQL
  question, not a document question. Route such questions to a structured data tool instead.

## Measuring success in production

- Retrieval recall@5 and MRR on a fixed test set, tracked on every change.
- Share of answers with at least one citation, and share of declined questions.
- User feedback (thumbs up or down) and the share of conversations that still end in a ticket.
- Latency (p50 and p95) and cost per answered question.

## Checklist

- Split by document structure and keep titles and metadata with every chunk.
- Run keyword and vector search in parallel and merge with RRF.
- Rerank the top candidates with a cross-encoder.
- Apply a minimum score and allow the system to decline.
- Require citations in every answer.
- Enforce access control in the retriever.
- Build a test set before tuning anything.

## Final thought

Language models get most of the attention, but in my experience the quality of a RAG assistant is
decided by retrieval and by the honesty of its "I don't know". Get those two right and users start to
trust it. If you are planning or rescuing an internal assistant, see my
[generative AI and RAG service](/services/#enterprise-generative-ai) or [get in touch](/contact/).
