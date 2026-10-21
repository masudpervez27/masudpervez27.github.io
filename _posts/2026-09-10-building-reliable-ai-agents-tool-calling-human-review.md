---
title: "Building AI Agents You Can Trust: Tool Calling, Explicit State and Human Approval"
date: 2026-09-10T10:00:00+02:00
excerpt: "An AI agent that can take actions is only useful if the business can control what it does. How to design agents as explicit state machines with typed tools, risk-based human approval and a full audit trail, with the reliability math, a modular Python implementation and a customer service example."
categories:
  - Generative AI
  - AI Agents
tags:
  - AI Agents
  - LangGraph
  - Tool Calling
  - Human-in-the-Loop
  - Pydantic
  - Python
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
toc_icon: "robot"
---

An assistant that answers questions can embarrass you. An agent that takes actions, like changing an
order, booking a slot or issuing a refund, can cost you money. That difference is why I design agents
very differently from chatbots. The model is allowed to *propose* steps, but the surrounding code
decides what is permitted, what needs a person to approve, and what gets logged. This post explains
that design, the simple probability math that shows why it matters, and a modular Python
implementation you can adapt.

## The business problem

Many processes are a sequence of look-ups and small decisions spread over several systems: check an
order, check a contract, check stock, update a record, write to the customer. People spend a large
part of their day moving information between screens. AI agents can automate much of that work, but
organisations hesitate for good reasons:

- **Uncontrolled actions.** A model that misunderstands a request might change the wrong record.
- **No accountability.** If something goes wrong, nobody can reconstruct why the agent did what it
  did.
- **Unpredictable cost and latency.** An agent stuck in a loop can call tools and models dozens of
  times for a single request.

The goal is an agent that does the routine work on its own, hands the risky parts to a person with all
the context prepared, and leaves a complete record. That is achievable today, but only with an
explicit design.

## The idea in plain English

I think of an agent as three layers:

1. **A planner** (the language model) that reads the request and proposes the next step: call a
   specific tool with specific arguments, or give a final answer.
2. **A controller** (ordinary code) that checks every proposal: is this tool allowed, are the
   arguments valid, how risky is the action, has the step budget run out?
3. **Tools** (ordinary functions) that do the actual work against your systems, each with a typed
   input and a declared risk level.

Low-risk read-only tools run automatically. Actions that change something run automatically only if
they are below a risk limit; otherwise the agent pauses and a person approves or rejects with one
click. Everything is recorded. The model is creative; the controller is strict.

<div class="diagram">
  <div class="dg-flow">
    <div class="dg-node">Request</div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Planner<small>language model proposes a step</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node dg-accent">Controller<small>validate, budget, risk</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-branch">
      <div class="dg-node dg-ok">Run tool automatically</div>
      <div class="dg-node dg-warn">Pause for human approval</div>
    </div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Your systems<small>orders, CRM, ERP</small></div>
    <div class="dg-arrow" aria-hidden="true"></div>
    <div class="dg-node">Answer and audit log</div>
  </div>
  <p class="dg-note">Tool results go back to the planner until it gives a final answer or the step budget runs out.</p>
  <p class="dg-caption">A creative planner inside a strict controller.</p>
</div>

## How it works: the math, step by step

### Why multi-step agents fail more than you expect

Suppose a task needs $$n$$ steps and each step is done correctly with probability $$p$$,
independently. The probability that the whole task succeeds is

$$
P(\text{success}) = p^{n}
$$

In plain English: errors compound. With $$p = 0.95$$ per step, a 3-step task succeeds about 86
percent of the time, but a 10-step task only about 60 percent of the time. This is the main reason I
keep agent workflows short and push as much as possible into deterministic code.

### What validation and retries buy you

If each step's output is checked by a validator that catches an error with probability $$v$$, and a
caught error is retried up to $$r$$ times, the effective success probability of one step becomes
approximately

$$
p_{\text{eff}} = 1 - (1 - p)\Big[(1 - v) + v\,(1 - p)^{r}\Big]
$$

Line by line:

- $$(1 - p)$$ is the chance the first attempt is wrong.
- $$(1 - v)$$ is the chance that the error slips past validation, so it stays wrong.
- $$v (1 - p)^{r}$$ is the chance it was caught but all $$r$$ retries also failed.

With $$p = 0.95$$, $$v = 0.8$$ and $$r = 2$$, this gives $$p_{\text{eff}} \approx 0.990$$, and the
10-step task now succeeds about 90 percent of the time instead of 60. Typed tool arguments and schema
checks are cheap ways to raise $$v$$.

### When should a person approve an action?

For an action $$a$$, let $$q_a$$ be the probability that it is wrong (estimated from history or a
confidence score), $$I_a$$ the business impact if a wrong action goes through, and $$C_h$$ the cost of
a human review. Sending the action for review is worthwhile when

$$
q_a \, I_a > C_h
$$

In plain English: ask a person when the expected damage of a mistake is larger than the cost of their
time. A refund of 50 SEK with a 2 percent error rate has an expected damage of 1 SEK, so it should not
wait for review that costs 30 SEK of staff time. A contract change with an impact of 50,000 SEK should
be reviewed even with a 1 percent error rate. This gives every tool a clear, defensible approval rule
instead of "the agent can do anything" or "a person approves everything".

### Bounding cost and latency

If each model call costs $$c_m$$ and takes $$t_m$$ seconds, and each tool call costs $$c_t$$ and takes
$$t_t$$, a hard limit of $$S$$ steps bounds the worst case:

$$
\text{Cost}_{\max} = S\,(c_m + c_t), \qquad \text{Latency}_{\max} = S\,(t_m + t_t)
$$

That is the reason for an explicit step budget: it turns an open-ended loop into a process with a
known maximum cost.

## Implementation in Python

The core is plain Python, which keeps it testable and independent of any agent framework. I then use
LangGraph for the part frameworks are genuinely good at: pausing for a person and resuming later.

### Tools with typed arguments and risk levels

```python
# agent/tools.py
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

from pydantic import BaseModel


class Risk(IntEnum):
    READ = 0      # no side effects
    LOW = 1       # reversible, small impact
    HIGH = 2      # money, contracts, customer-visible changes


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[BaseModel], dict]
    risk: Risk
    impact: Callable[[BaseModel], float] = lambda args: 0.0  # business impact if wrong


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name]

    def describe(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description}. Arguments schema: {t.args_model.model_json_schema()}"
            for t in self._tools.values()
        )
```

### Concrete tools for a delivery service

```python
# agent/delivery_tools.py
from pydantic import BaseModel, Field

from agent.tools import Risk, Tool


class OrderRef(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{6}$")


class AddressChange(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{6}$")
    street: str = Field(min_length=3, max_length=120)
    postcode: str = Field(pattern=r"^\d{3} ?\d{2}$")
    city: str = Field(min_length=2, max_length=60)


class Refund(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{6}$")
    amount_sek: float = Field(gt=0, le=10_000)
    reason: str = Field(min_length=5, max_length=300)


def build_tools(orders_api) -> list[Tool]:
    return [
        Tool("get_order", "Look up an order and its delivery status", OrderRef,
             lambda a: orders_api.get(a.order_id), Risk.READ),
        Tool("change_address", "Change the delivery address of an order not yet out for delivery",
             AddressChange, lambda a: orders_api.change_address(**a.model_dump()), Risk.LOW,
             impact=lambda a: 150.0),
        Tool("issue_refund", "Refund part or all of an order", Refund,
             lambda a: orders_api.refund(a.order_id, a.amount_sek, a.reason), Risk.HIGH,
             impact=lambda a: a.amount_sek),
    ]
```

The `Field` constraints are a large part of the validator $$v$$ from the math section: a malformed
order id or a refund above the limit is rejected before any system is touched.

### The controller

```python
# agent/controller.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pydantic import BaseModel, ValidationError

from agent.tools import Risk, ToolRegistry


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class Decision(BaseModel):
    kind: Literal["tool", "final"]
    tool: str | None = None
    args: dict = {}
    answer: str | None = None
    confidence: float = 0.5


@dataclass
class Policy:
    max_steps: int = 6
    max_retries: int = 2
    review_cost: float = 30.0
    auto_approve_max_risk: Risk = Risk.LOW


@dataclass
class AgentState:
    request: str
    history: list[dict] = field(default_factory=list)
    pending: dict | None = None
    answer: str | None = None
    status: Literal["running", "needs_approval", "done", "failed"] = "running"


SYSTEM = (
    "You are an operations assistant. Decide the next step. Respond with JSON only: "
    '{"kind": "tool", "tool": "<name>", "args": {...}, "confidence": 0.0-1.0} or '
    '{"kind": "final", "answer": "..."}. Use only the listed tools. Never invent order data.'
)


class AgentController:
    def __init__(self, llm: LLM, tools: ToolRegistry, policy: Policy, audit):
        self.llm, self.tools, self.policy, self.audit = llm, tools, policy, audit

    def _prompt(self, state: AgentState) -> str:
        return (f"Tools:\n{self.tools.describe()}\n\nRequest: {state.request}\n\n"
                f"Steps so far:\n{json.dumps(state.history, ensure_ascii=False)}")

    def _decide(self, state: AgentState) -> Decision:
        last_error = ""
        for _ in range(self.policy.max_retries + 1):
            raw = self.llm.complete(SYSTEM, self._prompt(state) + last_error)
            try:
                return Decision.model_validate_json(raw)
            except ValidationError as e:
                last_error = f"\n\nYour last reply was invalid: {e.errors()[:2]}. Reply with valid JSON."
        raise RuntimeError("Model did not return a valid decision")

    def needs_approval(self, tool, args, confidence: float) -> bool:
        if tool.risk <= Risk.READ:
            return False
        if tool.risk > self.policy.auto_approve_max_risk:
            return True
        expected_damage = (1 - confidence) * tool.impact(args)
        return expected_damage > self.policy.review_cost

    def step(self, state: AgentState) -> AgentState:
        if len(state.history) >= self.policy.max_steps:
            state.status = "failed"
            state.answer = "Step limit reached. Handing over to a colleague."
            return state

        decision = self._decide(state)
        if decision.kind == "final":
            state.answer, state.status = decision.answer, "done"
            self.audit.log("final", state.request, decision.model_dump())
            return state

        tool = self.tools.get(decision.tool)
        try:
            args = tool.args_model.model_validate(decision.args)
        except ValidationError as e:
            state.history.append({"tool": tool.name, "error": e.errors()[:2]})
            return state

        if self.needs_approval(tool, args, decision.confidence):
            state.pending = {"tool": tool.name, "args": args.model_dump(), "confidence": decision.confidence}
            state.status = "needs_approval"
            self.audit.log("approval_requested", state.request, state.pending)
            return state

        return self.execute(state, tool.name, args.model_dump())

    def execute(self, state: AgentState, tool_name: str, args: dict) -> AgentState:
        tool = self.tools.get(tool_name)
        started = time.perf_counter()
        result = tool.func(tool.args_model.model_validate(args))
        state.history.append({"tool": tool_name, "args": args, "result": result})
        self.audit.log("tool_call", state.request, {"tool": tool_name, "args": args,
                       "ms": round(1000 * (time.perf_counter() - started))})
        state.pending, state.status = None, "running"
        return state

    def run(self, state: AgentState) -> AgentState:
        while state.status == "running":
            state = self.step(state)
        return state
```

Three details carry most of the safety:

- The model never calls a function directly. It returns a `Decision`, and the controller decides.
- Tool arguments are validated against the tool's schema before anything runs.
- Approval is decided by code from risk, impact and confidence, not by the model.

### Pausing for a person with LangGraph

When an approval is needed, the request may wait minutes or hours. LangGraph's checkpointing and
`interrupt` make that straightforward: the graph saves its state, returns control, and resumes from
the same point when a decision arrives.

```python
# agent/graph.py
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.controller import AgentController, AgentState


class GraphState(TypedDict):
    agent: AgentState


def build_graph(controller: AgentController):
    def think(state: GraphState) -> dict:
        return {"agent": controller.step(state["agent"])}

    def approve(state: GraphState) -> dict:
        agent = state["agent"]
        decision = interrupt({"request": agent.request, "proposed_action": agent.pending})
        if decision == "approve":
            agent = controller.execute(agent, agent.pending["tool"], agent.pending["args"])
        else:
            agent.history.append({"tool": agent.pending["tool"], "rejected_by_human": True})
            agent.pending, agent.status = None, "running"
        return {"agent": agent}

    def route(state: GraphState) -> str:
        status = state["agent"].status
        return {"running": "think", "needs_approval": "approve"}.get(status, "end")

    g = StateGraph(GraphState)
    g.add_node("think", think)
    g.add_node("approve", approve)
    g.add_edge(START, "think")
    g.add_conditional_edges("think", route, {"think": "think", "approve": "approve", "end": END})
    g.add_conditional_edges("approve", route, {"think": "think", "approve": "approve", "end": END})
    return g.compile(checkpointer=MemorySaver())


# Usage
# app = build_graph(controller)
# cfg = {"configurable": {"thread_id": "ticket-48213"}}
# app.invoke({"agent": AgentState(request="...")}, cfg)      # pauses at approval
# app.invoke(Command(resume="approve"), cfg)                  # resumes after a click
```

In production, replace `MemorySaver` with a persistent checkpointer (for example PostgreSQL) so a
paused request survives restarts, and store the audit log in an append-only table.

## A worked example: delivery changes in customer service

A delivery company receives around 8,000 customer messages a week. About a third are variations of
"where is my parcel, and can I change the address?". Each takes an agent 4 to 6 minutes across three
systems.

With the design above:

- **"Where is ORD-482913?"** The agent calls `get_order` (read-only), and answers with the status. No
  approval, a few seconds.
- **"Please deliver ORD-482913 to Storgatan 12, 113 45 Stockholm instead."** The agent calls
  `get_order`, sees the parcel is still at the hub, proposes `change_address` with confidence 0.93.
  Expected damage is $$0.07 \times 150 \approx 10$$ SEK, below the 30 SEK review cost, so it runs
  automatically and the customer gets a confirmation.
- **"The parcel arrived damaged, I want my 1,200 SEK back."** The agent prepares `issue_refund` with
  the order, amount and reason. The risk level is high, so it pauses. A team member sees the request,
  the order details and the proposed action on one screen, and approves in seconds.

In a setup like this, most of the routine volume is handled end to end, the risky part is reviewed by
a person with everything prepared, and every action can be traced back to the request, the model's
proposal and the approval. That is the level of control that lets an operations manager actually
switch it on.

## Pitfalls and when not to use it

- **Letting the model call APIs directly.** Always put a controller with validation in between.
- **Giving agents broad credentials.** Each tool should use a service account with the minimum
  permissions it needs, ideally read-only for read tools.
- **Prompt injection through data.** Text in emails, documents or web pages can contain instructions.
  Treat tool results as data, never as instructions, and never let them change the tool list or the
  policy.
- **Long autonomous chains.** Remember $$p^{n}$$. If a process always follows the same steps, write
  it as normal code and use the model only for the parts that need language understanding.
- **No evaluation set.** Build a set of realistic requests with the expected tool sequence and
  outcome, and run it on every change of prompt, model or tools.
- **When rules are enough.** If the decision can be written as a small set of clear rules, a rules
  engine is cheaper, faster and easier to audit.

## Measuring success in production

- Share of requests completed without a person, and share escalated.
- Approval rate of proposed high-risk actions (a low rate means the agent proposes badly).
- Error rate found in audits, by tool.
- Average steps, latency and model cost per request, against the step budget.
- Handling time saved per request, which is the business case.

## Checklist

- Define each tool with a typed argument schema, a risk level and an impact estimate.
- Keep the model's role to proposing decisions in a strict JSON format.
- Validate every decision and every argument before execution.
- Decide approval in code using risk, impact and confidence.
- Set a step budget and a clear hand-over message.
- Persist state so approvals can pause and resume.
- Log every proposal, approval and tool call in an append-only audit trail.
- Test against a fixed set of realistic requests on every change.

## Final thought

The useful question is not "can an AI agent do this task?" but "can we control and verify what it
does?". Designing the agent as a strict controller around a creative model answers that question,
and in my experience it is what gets agents from a pilot into daily operations. If you are exploring
agents for your own processes, see my
[AI automation and agentic workflow service](/services/#ai-automation-agentic-workflow) or
[get in touch](/contact/).
