---
title: "Building Enterprise AI Securely and Responsibly: Develop, Govern, Deploy, Observe"
date: 2026-10-20T10:00:00+02:00
excerpt: "Most AI pilots never reach production, and the reason is rarely the model. It is security, governance and operations. A practical lifecycle for enterprise AI, with the math behind risk scoring, release gates and error budgets, modular Python building blocks, and an internal assistant taken from pilot to production."
categories:
  - Generative AI
  - MLOps
tags:
  - AI Governance
  - AI Security
  - EU AI Act
  - GDPR
  - LLMOps
  - Observability
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "shield-halved"
---

The pattern I see most often in AI projects is a successful pilot that never becomes a product. The
demo works and users like it, then the security review asks where the data goes, legal asks who is
accountable for wrong answers, IT operations asks how it will be monitored, and finance asks what it
will cost at full scale. Nobody has good answers, so the pilot stalls. In my experience the gap is
almost never the model. It is everything around it. This post describes the lifecycle I use to take
AI systems into production in a way that security, legal and operations can sign off on: develop,
govern, deploy and observe, with the math behind the key decisions and Python building blocks for
each stage.

## The business problem

Enterprise AI fails in predictable ways:

- **Security incidents.** Prompt injection, leaking confidential documents to the wrong users, or an
  assistant that can be talked into calling tools it should not use.
- **Compliance gaps.** Personal data sent to a provider without a data processing agreement, no data
  protection impact assessment, no record of what the system does, and no plan for the EU AI Act.
- **Quality nobody measures.** Releases are approved because "it looked good", and quality drops
  silently after a model update.
- **Unplanned cost and downtime.** Token costs grow faster than usage, and when the system is slow
  or down, nobody notices until users complain.

Each of these is manageable on its own. Together, and discovered late, they kill projects. Building
the controls in from the start is cheaper than retrofitting them, and it is what turns AI from an
experiment into something the organisation can depend on.

## The idea in plain English

I split the lifecycle into four stages, each with a clear owner and a clear question:

1. **Develop.** *Is it good and safe enough?* Build with real data under control, test against a
   fixed evaluation set and a set of attacks, and treat prompts and configuration as code.
2. **Govern.** *Should we run it, and under what conditions?* Record the system, assess its risks,
   check the legal basis, decide who is accountable, and define where people stay in the loop.
3. **Deploy.** *Can we release it without surprises?* Automated release gates, staged roll-out,
   least-privilege access and a tested rollback.
4. **Observe.** *Is it still working, safe and affordable?* Measure quality, safety, latency, cost
   and drift continuously, keep an audit trail, and have a plan for incidents.

Across all four run the same controls: access management, data protection, logging and clear
ownership. None of this is exotic. It is the same discipline used for any critical system, adapted to
the specific risks of models that learn from data and generate text.

<div class="diagram">
  <div class="dg-stages">
    <div class="dg-stage">
      <div class="dg-stage-title">1. Develop</div>
      <ul>
        <li>Data contracts and minimisation</li>
        <li>Evaluation set as a release gate</li>
        <li>Threat model and red-team tests</li>
        <li>Prompts and config in version control</li>
      </ul>
    </div>
    <div class="dg-stage">
      <div class="dg-stage-title">2. Govern</div>
      <ul>
        <li>AI system register and owner</li>
        <li>Risk matrix and residual risk</li>
        <li>DPIA and EU AI Act classification</li>
        <li>Human oversight rules</li>
      </ul>
    </div>
    <div class="dg-stage">
      <div class="dg-stage-title">3. Deploy</div>
      <ul>
        <li>CI/CD with quality and safety gates</li>
        <li>Shadow or canary release</li>
        <li>Least-privilege service accounts</li>
        <li>Tested rollback</li>
      </ul>
    </div>
    <div class="dg-stage">
      <div class="dg-stage-title">4. Observe</div>
      <ul>
        <li>Quality, safety and drift metrics</li>
        <li>Latency SLOs and error budgets</li>
        <li>Cost per request and budgets</li>
        <li>Audit trail and incident process</li>
      </ul>
    </div>
  </div>
  <div class="dg-bar">Across all stages: identity and access control, data protection, logging, clear ownership</div>
  <p class="dg-caption">A lifecycle for enterprise AI that security, legal and operations can sign off on.</p>
</div>

## How it works: the math, step by step

Most of this lifecycle is process, but four decisions benefit from simple, explicit math. Making them
numerical turns opinions ("it seems safe enough") into thresholds that can be agreed in advance.

### 1. Risk scoring and residual risk

For each identified risk, score likelihood $$L$$ and impact $$I$$ on a scale from 1 to 5:

$$
R = L \times I
$$

In plain English: a risk that is both likely and severe scores up to 25; a rare and minor one scores
1. After adding controls (for example permission-filtered retrieval), score the **residual** risk
with the new likelihood $$L'$$ and impact $$I'$$:

$$
R' = L' \times I'
$$

The organisation agrees an acceptance level $$R_{\max}$$ in advance, for example 6. A system goes
live only when every residual risk satisfies $$R' \le R_{\max}$$, or a named owner formally accepts
the exception. The value of this is not precision. It is that the decision and its reasons are written
down and comparable across systems.

### 2. A release gate that accounts for test set size

Suppose the evaluation set has $$n$$ questions and the system answers $$k$$ of them correctly, so the
observed accuracy is $$\hat{p} = k / n$$. With a small test set, $$\hat{p}$$ can look good by luck. I
therefore gate on the **lower confidence bound** of accuracy, using the Wilson interval:

$$
\text{LB} = \frac{\hat{p} + \frac{z^2}{2n} - z \sqrt{\frac{\hat{p}(1 - \hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
$$

Line by line:

- $$\hat{p}$$ is the measured accuracy.
- The square-root term is the uncertainty, which shrinks as $$n$$ grows.
- $$z$$ sets the confidence level; $$z = 1.645$$ gives a one-sided 95 percent bound.
- The release passes only if $$\text{LB} \ge p_{\text{target}}$$.

In plain English: we only release when we are fairly sure the true accuracy is above the target, not
just when the test happened to look good. A larger test set gives a tighter bound and makes it easier
to pass with the same measured accuracy, which is a good incentive to invest in evaluation.

### 3. Service level objectives and error budgets

A service level objective (SLO) states the share of requests that must succeed, for example 99.5
percent over 30 days. The **error budget** is what is allowed to fail:

$$
\text{Budget} = (1 - \text{SLO}) \times N_{\text{requests}}
$$

The **burn rate** says how fast the budget is being used:

$$
\text{Burn rate} = \frac{\text{observed error rate}}{1 - \text{SLO}}
$$

Read it as: a burn rate of 1 uses the budget exactly over the 30-day window; a burn rate of 10 would
use it in 3 days. A common alerting rule from site reliability engineering pages someone when the
burn rate over the last hour exceeds 14.4, because

$$
14.4 \times \frac{1 \text{ hour}}{720 \text{ hours}} = 0.02
$$

which means 2 percent of the monthly budget has gone in one hour. A slower burn (for example above 6
over six hours) opens a ticket instead of waking someone up. For AI systems I define SLOs not only for
availability and latency, but also for quality, for example "at least 95 percent of sampled answers
have a valid citation".

### 4. Cost per request

For a language model priced per million tokens, the cost of one request with $$t_{\text{in}}$$ input
tokens and $$t_{\text{out}}$$ output tokens is

$$
C = \frac{t_{\text{in}} \, p_{\text{in}} + t_{\text{out}} \, p_{\text{out}}}{10^6}
$$

where $$p_{\text{in}}$$ and $$p_{\text{out}}$$ are the prices per million tokens. In plain English:
cost grows with how much context you send and how long the answers are. Monthly cost is this times the
number of requests. Input tokens usually dominate in retrieval-based systems, so the cheapest
optimisation is often sending fewer, better passages, not switching models.

## Implementation in Python

The building blocks below are deliberately small. They are meant to be wired into your existing CI/CD,
logging and monitoring stack rather than replace it.

### Risk register

```python
# governance/risk.py
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Risk:
    risk_id: str
    description: str
    likelihood: int            # 1-5 before controls
    impact: int                # 1-5 before controls
    controls: list[str] = field(default_factory=list)
    residual_likelihood: int | None = None
    residual_impact: int | None = None
    owner: str = ""
    accepted_exception: bool = False

    @property
    def inherent(self) -> int:
        return self.likelihood * self.impact

    @property
    def residual(self) -> int:
        lik = self.residual_likelihood or self.likelihood
        imp = self.residual_impact or self.impact
        return lik * imp


@dataclass
class RiskRegister:
    system: str
    max_residual: int = 6
    risks: list[Risk] = field(default_factory=list)

    def blocking(self) -> list[Risk]:
        return [r for r in self.risks if r.residual > self.max_residual and not r.accepted_exception]

    def ready_for_production(self) -> bool:
        return not self.blocking() and all(r.owner for r in self.risks)
```

### Release gate

```python
# deploy/release_gate.py
from __future__ import annotations

import math
from dataclasses import dataclass, field

from governance.risk import RiskRegister


def wilson_lower_bound(successes: int, n: int, z: float = 1.645) -> float:
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (centre - margin) / denom


@dataclass
class EvalResult:
    name: str
    passed: int
    total: int
    target: float


@dataclass
class ReleaseGate:
    evals: list[EvalResult]
    risks: RiskRegister
    p95_latency_ms: float
    max_p95_latency_ms: float
    cost_per_request: float
    max_cost_per_request: float
    sign_offs: dict[str, bool] = field(default_factory=dict)  # e.g. {"security": True, "dpo": True}

    def check(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for e in self.evals:
            lb = wilson_lower_bound(e.passed, e.total)
            if lb < e.target:
                reasons.append(f"{e.name}: lower bound {lb:.3f} below target {e.target:.3f}")
        if not self.risks.ready_for_production():
            reasons.append(f"risks above limit: {[r.risk_id for r in self.risks.blocking()]}")
        if self.p95_latency_ms > self.max_p95_latency_ms:
            reasons.append(f"p95 latency {self.p95_latency_ms:.0f} ms above limit")
        if self.cost_per_request > self.max_cost_per_request:
            reasons.append(f"cost per request {self.cost_per_request:.4f} above limit")
        missing = [k for k, ok in self.sign_offs.items() if not ok]
        if missing:
            reasons.append(f"missing sign-off: {missing}")
        return (not reasons), reasons
```

In CI, the pipeline runs the evaluation suites (quality, safety and red-team prompts), builds a
`ReleaseGate`, and fails the build with the list of reasons if `check()` returns false. The reasons
are written to the release record, so every deployment has a documented basis.

### Audit logging without storing more personal data than needed

```python
# observe/audit.py
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid


class AuditLogger:
    """Append-only JSON lines. User ids are pseudonymised with a keyed hash."""

    def __init__(self, sink, system: str, model_version: str):
        key = os.environ.get("AUDIT_HMAC_KEY")
        if not key:
            raise RuntimeError("AUDIT_HMAC_KEY is not set")
        self._key = key.encode()
        self.sink, self.system, self.model_version = sink, system, model_version

    def _pseudonym(self, user_id: str) -> str:
        return hmac.new(self._key, user_id.encode(), hashlib.sha256).hexdigest()[:24]

    def log(self, user_id: str, event: str, **fields) -> str:
        record = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "system": self.system,
            "model_version": self.model_version,
            "user": self._pseudonym(user_id),
            "event": event,
            **fields,
        }
        self.sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record["id"]
```

I log identifiers of retrieved documents, tool calls, the policy decision and a hash of the prompt,
rather than full prompts by default. Full text logging is switched on only where the risk assessment
says it is needed, and with a defined retention period.

### Error budget burn alerts

```python
# observe/slo.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SLO:
    target: float = 0.995
    page_burn: float = 14.4   # over the last hour
    ticket_burn: float = 6.0  # over the last six hours


def burn_rate(errors: int, requests: int, slo: SLO) -> float:
    if requests == 0:
        return 0.0
    return (errors / requests) / (1 - slo.target)


def alert_level(err_1h: int, req_1h: int, err_6h: int, req_6h: int, slo: SLO = SLO()) -> str:
    if burn_rate(err_1h, req_1h, slo) > slo.page_burn:
        return "page"
    if burn_rate(err_6h, req_6h, slo) > slo.ticket_burn:
        return "ticket"
    return "ok"
```

"Errors" here should include quality failures that users experience, such as timeouts, refusals
caused by system faults, and answers that fail automated grounding checks, not only HTTP 500s.

### A thin input guard

```python
# develop/guards.py
from __future__ import annotations

import re

PERSONNUMMER = re.compile(r"\b(?:\d{2})?\d{6}[-+]?\d{4}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
MAX_CHARS = 4_000


def redact(text: str) -> str:
    text = PERSONNUMMER.sub("[PERSONNUMMER]", text)
    return EMAIL.sub("[EMAIL]", text)


def check_input(text: str) -> tuple[bool, str]:
    if len(text) > MAX_CHARS:
        return False, "input too long"
    return True, redact(text)
```

This is a first line of defence, not a security boundary. The real protections against prompt
injection and data leakage are architectural: permission-filtered retrieval, tools with least
privilege, validated tool arguments, and human approval for high-impact actions (see my post on
[building AI agents you can trust](/posts/2026-09-10-building-reliable-ai-agents-tool-calling-human-review/)).

## The regulations and standards to know

You do not need to be a lawyer to build responsibly, but you do need to know which frameworks apply
and involve the right people early:

- **GDPR.** A legal basis for processing, data minimisation, a data processing agreement with every
  provider that handles personal data, and a data protection impact assessment (DPIA) when the
  processing is likely to be high risk. In Sweden, the supervisory authority is IMY.
- **EU AI Act.** It entered into force in August 2024 and applies in stages. Some obligations already
  apply, such as the bans on prohibited practices and the requirement for AI literacy among staff;
  rules for general-purpose AI models followed in 2025; and requirements for high-risk systems are
  being phased in. Classify each system by risk level and check the current timeline, since dates
  have been discussed for adjustment.
- **NIST AI Risk Management Framework.** A practical structure (Govern, Map, Measure, Manage) that
  maps well onto the lifecycle in this post.
- **ISO/IEC 42001.** A management system standard for AI, useful if your organisation already works
  with ISO 27001 and wants a certifiable approach.
- **OWASP Top 10 for LLM Applications.** A checklist of the most common security risks, including
  prompt injection, sensitive information disclosure, excessive agency and unbounded consumption.
  I use it directly as input to the threat model.

## A worked example: an internal policy assistant from pilot to production

A Swedish company with 900 employees has piloted an assistant that answers questions about internal
policies (HR, travel, IT security, procurement). Twenty pilot users love it. Here is how it gets to
production.

**Develop.** The team builds an evaluation set of 500 real questions from the service desk, plus 80
red-team prompts (attempts to extract confidential HR documents, override instructions, or get
answers outside the policy scope). On the latest version, 460 of 500 answers are correct and cited.

$$
\hat{p} = 0.92, \qquad \text{LB} \approx 0.898
$$

The agreed target is 0.88, so the quality gate passes. A target of 0.90 would not have passed, and
the team would have needed either a better system or a larger test set to prove it. All 80 red-team
prompts are handled safely.

**Govern.** The risk register includes "assistant reveals HR case documents to unauthorised staff",
scored at likelihood 3 and impact 5, so $$R = 15$$. With retrieval filtered by the user's existing
document permissions, and red-team tests in every release, the residual likelihood is 1:
$$R' = 1 \times 5 = 5 \le 6$$. The risk is accepted with the HR director as owner. A DPIA is completed
with the data protection officer, the provider's data processing agreement is signed, and the system
is classified as limited risk under the EU AI Act, with a clear notice to users that they are talking
to an AI system.

**Deploy.** The release gate runs in CI. The assistant is first deployed in shadow mode for one week,
answering in the background while the service desk answers as usual, and the two are compared.
Then it is rolled out to one department, then everyone. The service account can only read the policy
library. Rollback to the previous version has been tested.

**Observe.** With about 3,000 questions a day (90,000 a month) and an SLO of 99.5 percent, the
monthly error budget is $$0.005 \times 90{,}000 = 450$$ failed requests. One afternoon, a provider
problem causes 8 percent of requests to fail for an hour. The one-hour burn rate is
$$0.08 / 0.005 = 16$$, above the paging level of 14.4, so the on-call engineer is paged and switches
to the fallback model. The incident itself uses only about 10 of the 450 allowed failures (8 percent
of roughly 125 requests in that hour). That is why burn rate, not a raw count, is the right trigger:
it catches a fast failure early, while it is still cheap.

Cost is tracked per request. With about 3,000 input tokens and 400 output tokens per question, and
illustrative prices of 2.5 and 10 (in USD per million tokens), each question costs

$$
C = \frac{3{,}000 \times 2.5 + 400 \times 10}{10^6} = 0.0115 \text{ USD}
$$

or about 35 USD a day. That is small next to the service desk time saved, and it is now a known,
monitored number rather than a surprise on the invoice.

The result: security, legal, IT operations and finance each have the evidence they need, and the
assistant moves from a pilot to a supported service.

## Pitfalls and when not to use it

- **Governance as paperwork only.** A risk register nobody reads does not reduce risk. Tie it to the
  release gate so it blocks releases when it should.
- **Testing once.** Model providers update models, documents change, users find new ways to ask.
  Re-run evaluations and red-team tests on every change and on a schedule.
- **Logging everything.** Full prompt logs are a data protection risk of their own. Log what you need
  for audit and debugging, pseudonymise, and set retention.
- **Security by prompt.** "Never reveal confidential information" in a system prompt is not a
  control. Enforce access in the retrieval and tool layers.
- **One SLO for availability only.** An assistant can be up and still be wrong. Add quality SLOs.
- **Overbuilding for low-risk use.** A small internal summarisation tool does not need the same
  process as a customer-facing decision system. Scale the controls to the risk.

## Measuring success in production

- Time from pilot to production, and share of pilots that make it.
- Release gate pass rate and the main reasons for failures.
- Quality SLO attainment (for example share of answers with valid citations).
- Security findings from red-team tests and from production, by severity.
- Availability, p95 latency and error budget remaining each month.
- Cost per request and per active user, against budget.

## Checklist

- Build an evaluation set and a red-team set before the first release.
- Keep prompts, configuration and evaluation thresholds in version control.
- Record each AI system with an owner, purpose and risk classification.
- Score risks, add controls, and agree a residual risk limit.
- Complete a DPIA where personal data is involved, and sign data processing agreements.
- Gate releases on the lower confidence bound of quality, not the raw score.
- Use least-privilege service accounts and permission-filtered retrieval.
- Release in shadow or canary mode, with a tested rollback.
- Define SLOs for availability, latency and quality, and alert on burn rate.
- Log for audit with pseudonymised identifiers and defined retention.

## Final thought

Responsible AI is not a separate project that slows delivery down. Done well, it is what makes
delivery possible, because it gives every stakeholder the evidence they need to say yes. The math
involved is simple; the value is in agreeing the thresholds before anyone is under pressure to ship.
If you want help taking an AI pilot to production, or setting up governance and observability for
the systems you already run, see my
[cloud data and AI platform service](/services/#cloud-data-ai-platform) or
[contact me](/contact/).
