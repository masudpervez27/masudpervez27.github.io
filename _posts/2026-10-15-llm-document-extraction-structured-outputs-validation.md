---
title: "Turning Documents into Data: LLM Extraction with Structured Outputs, Validation and Confidence Routing"
date: 2026-10-15T10:00:00+02:00
excerpt: "Invoices, delivery notes, contracts and forms still arrive as PDFs and get retyped by hand. How to extract them reliably with a language model: strict schemas, business-rule validation, a confidence score built from self-consistency, and the math for deciding which documents a person should check."
categories:
  - Generative AI
  - Automation
tags:
  - Document AI
  - LLM
  - Structured Outputs
  - Pydantic
  - Process Automation
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "file-invoice"
---

A surprising amount of business data still starts its life as a PDF: supplier invoices, delivery
notes, order confirmations, certificates, application forms. Somebody reads them and types the
important fields into a system. Language models are now good enough to do most of that reading, but
"most" is the problem. A system that is right 95 percent of the time and silently wrong the other 5
percent is worse than useless in finance or logistics. This post shows how I build extraction
pipelines that know when they are unsure: strict output schemas, validation against business rules,
a confidence score, and a cost-based rule for which documents go to a person.

## The business problem

Manual document handling has three costs that grow with volume:

- **Labour.** A few minutes per document across thousands of documents a month adds up to several
  full-time roles doing repetitive work.
- **Errors.** Typing mistakes lead to wrong payments, duplicate payments, missed discounts and
  reconciliation work later.
- **Delay.** Documents wait in a queue, so payments are late, early-payment discounts are lost and
  goods sit in the warehouse waiting for paperwork.

Older template-based OCR tools only work when every supplier uses the same layout, which they never
do. Language models handle layout variation well, but they need guardrails, because they will happily
produce a plausible number when the real one is unreadable.

## The idea in plain English

The pipeline has five steps:

1. **Read** the document into text (with layout) using OCR or a PDF parser.
2. **Extract** the fields with a language model that must return data in a fixed schema: supplier,
   invoice number, dates, line items, totals.
3. **Validate** the result with ordinary code: do the line items add up to the total? Is the VAT
   amount consistent with the rate? Is the due date after the invoice date? Does the supplier exist
   in our register?
4. **Score confidence** by asking the model more than once and checking whether it gives the same
   answer, combined with the validation results.
5. **Route** each document: confident and valid documents go straight into the system; the rest go
   to a person, with the fields pre-filled and the doubtful ones highlighted.

The person only checks what the machine is unsure about, and the threshold for "unsure" is set from
the cost of a review versus the cost of an error.

## How it works: the math, step by step

### Measuring extraction quality per field

For a field $$f$$ (for example "total amount") over a labelled test set, let TP be correct
extractions, FP be wrong or invented values, and FN be values that existed but were not extracted:

$$
\text{Precision}_f = \frac{TP}{TP + FP}, \qquad
\text{Recall}_f = \frac{TP}{TP + FN}, \qquad
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

In plain English: precision is "when it gave a value, how often was it right", recall is "of the
values that were there, how many did it find", and $$F_1$$ balances the two. For money fields I care
most about precision, because a wrong amount is worse than a missing one.

### Confidence from self-consistency

Language models do not give reliable probabilities for their own outputs. A practical substitute is
to run the extraction $$k$$ times (with some sampling randomness) and measure agreement. For field
$$f$$ with extracted values $$v_1, \dots, v_k$$:

$$
\text{conf}_f = \max_{v} \frac{1}{k} \sum_{j=1}^{k} \mathbf{1}[v_j = v]
$$

Read it as: the share of runs that agree with the most common answer. If all five runs read the
total as 12,480.00, confidence is 1.0. If three say 12,480.00 and two say 12,430.00, confidence is
0.6, which usually means a smudged digit.

The document confidence is driven by its weakest required field:

$$
\text{conf}_{\text{doc}} = \mathbf{1}[\text{all validations pass}] \cdot \min_{f \in \text{required}} \text{conf}_f
$$

In plain English: a document is only as trustworthy as its least certain important field, and any
failed business rule sets confidence to zero, sending it to a person.

### Choosing the routing threshold

Let $$t$$ be the threshold: documents with confidence at least $$t$$ are processed automatically.
Let $$C_r$$ be the cost of a human review and $$C_e$$ the cost of an error that slips through. The
expected cost per document is

$$
\text{Cost}(t) = C_r \cdot P(\text{conf} < t) + C_e \cdot P(\text{error},\ \text{conf} \ge t)
$$

Line by line:

- The first term is what we pay for reviews: the review cost times the share of documents below the
  threshold.
- The second term is what we pay for mistakes: the error cost times the share of documents that are
  wrong *and* were trusted.
- Raising $$t$$ increases the first term and lowers the second. We estimate both probabilities on a
  labelled validation set and choose the $$t$$ with the lowest total.

The **straight-through processing rate** is $$P(\text{conf} \ge t)$$, the share of documents nobody
touches. It is the number the business will ask about, and it should always be reported together
with the error rate among those documents.

## Implementation in Python

### The schema

```python
# extraction/schema.py
from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field, field_validator, model_validator


class LineItem(BaseModel):
    description: str = Field(min_length=1, max_length=300)
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    amount: Decimal = Field(ge=0)


class Invoice(BaseModel):
    supplier_name: str = Field(min_length=2, max_length=200)
    supplier_vat_id: str | None = None
    invoice_number: str = Field(min_length=1, max_length=50)
    invoice_date: date
    due_date: date | None = None
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    net_total: Decimal = Field(ge=0)
    vat_total: Decimal = Field(ge=0)
    gross_total: Decimal = Field(ge=0)
    lines: list[LineItem] = Field(default_factory=list)

    @field_validator("supplier_vat_id")
    @classmethod
    def normalise_vat_id(cls, v: str | None) -> str | None:
        return v.replace(" ", "").upper() if v else v

    @model_validator(mode="after")
    def dates_in_order(self) -> "Invoice":
        if self.due_date and self.due_date < self.invoice_date:
            raise ValueError("due_date is before invoice_date")
        return self
```

Using `Decimal` instead of `float` avoids rounding surprises in money fields. Schema rules that are
always true (formats, ranges, date order) belong here, so a malformed answer fails immediately.

### Business-rule validation

```python
# extraction/validation.py
from __future__ import annotations

from decimal import Decimal

from extraction.schema import Invoice

TOLERANCE = Decimal("0.05")


def validate_invoice(inv: Invoice, known_suppliers: set[str]) -> list[str]:
    """Rules that depend on arithmetic or on our own master data."""
    problems: list[str] = []
    if inv.lines:
        line_sum = sum(l.amount for l in inv.lines)
        if abs(line_sum - inv.net_total) > TOLERANCE:
            problems.append(f"lines sum {line_sum} != net_total {inv.net_total}")
        for i, l in enumerate(inv.lines):
            if abs(l.quantity * l.unit_price - l.amount) > TOLERANCE:
                problems.append(f"line {i}: quantity x price != amount")
    if abs(inv.net_total + inv.vat_total - inv.gross_total) > TOLERANCE:
        problems.append("net + vat != gross")
    if inv.supplier_vat_id and inv.supplier_vat_id not in known_suppliers:
        problems.append("unknown supplier VAT id")
    return problems
```

### Extraction with retries

```python
# extraction/extract.py
from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from extraction.schema import Invoice


class LLM(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str: ...


SYSTEM = (
    "Extract the invoice into JSON that matches this schema exactly. "
    "Use null for fields that are not present. Never guess numbers that are not readable. "
    "Dates as YYYY-MM-DD, amounts as plain numbers with a dot as decimal separator.\n"
    f"Schema: {Invoice.model_json_schema()}"
)


def extract_once(llm: LLM, document_text: str, temperature: float, max_retries: int = 2) -> Invoice | None:
    feedback = ""
    for _ in range(max_retries + 1):
        raw = llm.complete(SYSTEM, document_text + feedback, temperature=temperature)
        try:
            return Invoice.model_validate_json(raw)
        except ValidationError as e:
            feedback = f"\n\nYour previous output was invalid: {e.errors()[:3]}. Return corrected JSON only."
    return None
```

Feeding the validation error back to the model fixes most formatting mistakes on the second try.

### Self-consistency and confidence

```python
# extraction/confidence.py
from __future__ import annotations

from collections import Counter

from extraction.extract import LLM, extract_once
from extraction.schema import Invoice

REQUIRED = ("supplier_name", "invoice_number", "invoice_date", "currency", "net_total", "vat_total", "gross_total")


def extract_with_confidence(llm: LLM, text: str, k: int = 5) -> tuple[Invoice | None, dict[str, float]]:
    runs = [extract_once(llm, text, temperature=0.0 if i == 0 else 0.7) for i in range(k)]
    valid = [r for r in runs if r is not None]
    if not valid:
        return None, {f: 0.0 for f in REQUIRED}

    confidence: dict[str, float] = {}
    consensus: dict = {}
    for f in REQUIRED:
        values = Counter(str(getattr(r, f)) for r in valid)
        value, count = values.most_common(1)[0]
        confidence[f] = count / k  # failed runs count as disagreement
        consensus[f] = value

    # take the first valid run whose required fields match the consensus
    best = next((r for r in valid if all(str(getattr(r, f)) == consensus[f] for f in REQUIRED)), valid[0])
    return best, confidence
```

### Routing and threshold selection

```python
# extraction/routing.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RoutingCosts:
    review: float   # cost of one human review
    error: float    # cost of one undetected error


def document_confidence(field_conf: dict[str, float], problems: list[str]) -> float:
    return 0.0 if problems else min(field_conf.values())


def best_threshold(conf: np.ndarray, is_error: np.ndarray, costs: RoutingCosts) -> dict[str, float]:
    """conf: document confidence on a labelled set; is_error: 1 if the extraction was wrong."""
    best = None
    for t in np.unique(np.concatenate([conf, [1.01]])):
        auto = conf >= t
        cost = costs.review * np.mean(~auto) + costs.error * np.mean(auto & (is_error == 1))
        if best is None or cost < best["cost_per_doc"]:
            best = {"threshold": float(t), "cost_per_doc": float(cost),
                    "straight_through": float(auto.mean()),
                    "error_rate_auto": float(is_error[auto].mean()) if auto.any() else 0.0}
    return best
```

The threshold candidates include 1.01, which means "review everything". If that turns out to be the
cheapest option, the pipeline is not ready for automation yet, and the numbers say so clearly.

### The pipeline

```python
# extraction/pipeline.py
from extraction.confidence import extract_with_confidence
from extraction.routing import document_confidence
from extraction.validation import validate_invoice


def process(llm, text: str, known_suppliers: set[str], threshold: float) -> dict:
    invoice, field_conf = extract_with_confidence(llm, text)
    if invoice is None:
        return {"route": "manual", "reason": "extraction failed"}
    problems = validate_invoice(invoice, known_suppliers)
    conf = document_confidence(field_conf, problems)
    route = "auto" if conf >= threshold else "review"
    return {"route": route, "confidence": conf, "problems": problems,
            "low_confidence_fields": [f for f, c in field_conf.items() if c < 1.0],
            "invoice": invoice.model_dump(mode="json")}
```

The `low_confidence_fields` list is what makes reviews fast: the reviewer's screen highlights exactly
which fields to check instead of the whole document.

## A worked example: supplier invoices in accounts payable

A wholesale distributor receives about 4,000 supplier invoices a month from 600 suppliers. Manual
entry takes around 5 minutes per invoice, and the finance team estimates that an undetected error
costs on average 1,500 SEK in correction work, supplier calls and occasional overpayment. A review of
a pre-filled invoice takes about 1 minute, roughly 10 SEK.

On a labelled set of 500 past invoices:

- With confidence threshold 1.0 (all five runs agree and all rules pass), 74 percent of invoices go
  straight through, with an error rate among them of 0.3 percent.
- Expected cost per invoice is about $$10 \times 0.26 + 1{,}500 \times 0.74 \times 0.003 \approx 2.6 + 3.3 = 5.9$$ SEK.
- A lower threshold of 0.8 raises straight-through to 85 percent, but the error rate among automated
  invoices rises to 1.5 percent, and the expected cost climbs to about
  $$10 \times 0.15 + 1{,}500 \times 0.85 \times 0.015 \approx 1.5 + 19.1 = 20.6$$ SEK.

So the stricter threshold wins clearly, even though it automates fewer documents. That is the kind of
result that is hard to see without the cost formula, because "85 percent automated" sounds better in
a presentation than "74 percent".

In time terms, the team goes from about 330 hours a month of typing to around 20 hours of reviewing
pre-filled invoices, plus the time saved on correcting errors later. Exact results depend on document
quality and supplier mix, which is why the pilot always starts with a labelled sample.

## Pitfalls and when not to use it

- **Trusting a single run.** One extraction gives no measure of uncertainty. Use self-consistency or
  another independent signal.
- **Skipping arithmetic checks.** Totals, VAT and line sums catch a large share of reading errors at
  almost no cost.
- **Sending sensitive documents anywhere.** Check where the model runs and where data is stored. For
  personal data, use a provider and region that meet your GDPR obligations, or a self-hosted model.
- **Poor scans.** Very low quality images defeat any model. Detect them early (OCR confidence, page
  resolution) and route them to manual handling.
- **Prompt injection in documents.** A document can contain text like "ignore previous instructions".
  Extraction prompts should treat document content strictly as data, and validation should never be
  delegated to the model.
- **When layouts never change.** If 90 percent of your volume comes from three suppliers with fixed
  layouts, classic template extraction or e-invoicing (for example Peppol) may be cheaper and more
  reliable.

## Measuring success in production

- Straight-through processing rate and error rate among automated documents, reported together.
- Field-level precision for money and identifier fields.
- Average review time per routed document.
- Share of reviewer corrections by field, which tells you where to improve prompts or validation.
- End-to-end processing time from receipt to posting.

## Checklist

- Define a strict schema with types, formats and ranges.
- Validate arithmetic and master data in code, not in the prompt.
- Measure confidence with self-consistency on required fields.
- Label a few hundred real documents before choosing a threshold.
- Choose the threshold from review cost and error cost.
- Show reviewers only the doubtful fields.
- Monitor error rates among automated documents continuously.

## Final thought

The value of document automation is not in how many documents a model can read. It is in how
precisely the system knows which documents it has read correctly. Build that knowledge in with
schemas, validation and a cost-based routing rule, and you get automation the finance team can trust.
If you have a document-heavy process you would like to automate, see my
[AI automation and agentic workflow service](/services/#ai-automation-agentic-workflow) or
[contact me](/contact/).
