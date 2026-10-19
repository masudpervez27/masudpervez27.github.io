---
title: "Case Studies & Enterprise Projects"
layout: single
permalink: /projects/
author_profile: true
classes: wide
---

<div class="hero-tagline" style="margin-bottom: 1.5rem;">
  <i class="fas fa-chart-line"></i> Proven Enterprise Track Record &bull; Measurable Business Outcomes
</div>

A selection of mission-critical machine learning, distributed big data, and statistical optimization initiatives delivered for leading European institutions.

---

<div class="case-study-card">
  <div class="case-header">
    <span class="client-tag">Banking &bull; SEB (Skandinaviska Enskild Banken)</span>
    <span class="role-tag">Senior Data Scientist | Stockholm, Sweden</span>
  </div>
  <h2 class="case-title">Real-Time Fraud Detection Engine &amp; Cost-Based Threshold Optimization</h2>
  
  <p><strong>The Challenge:</strong> High-volume financial transactions required instantaneous risk scoring to intercept fraudulent operations without delaying genuine client payments or generating costly false alarms that overload fraud investigation desks.</p>
  
  <p><strong>The Solution Architecture:</strong></p>
  <ul>
    <li>Led the architectural design and deployment of real-time machine learning models integrated with bank microservices via low-latency REST/gRPC endpoints.</li>
    <li>Developed a <em>cost-based threshold tuning framework</em> that dynamically balances expected fraud loss against genuine customer friction costs, maximizing overall net savings.</li>
    <li>Constructed modular Apache Spark pipelines on Hadoop clusters for distributed feature extraction, large-scale historical transaction processing, and model evaluation.</li>
    <li>Established continuous model governance, drift detection, and automated recalibration protocols to counteract adversarial drift.</li>
    <li>Evaluated external fraud prevention vendors (RFP) to augment the bank's core financial crime prevention ecosystem.</li>
  </ul>

  <div class="impact-callout">
    <strong>Measurable Business Impact:</strong>
    <p>Achieved over <strong>60%–70% fraud detection recall</strong> in live production while preserving extremely low false-positive rates, significantly curtailing fraudulent transaction losses across Scandinavian accounts.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">Python</span>
    <span class="tech-pill">Apache Spark</span>
    <span class="tech-pill">Scala</span>
    <span class="tech-pill">Hadoop/HDFS</span>
    <span class="tech-pill">Scikit-Learn</span>
    <span class="tech-pill">Microservices</span>
    <span class="tech-pill">Model Governance</span>
  </div>
</div>

---

<div class="case-study-card">
  <div class="case-header">
    <span class="client-tag">Media &amp; Publishing &bull; Bonnier News</span>
    <span class="role-tag">Data Scientist | Stockholm, Sweden</span>
  </div>
  <h2 class="case-title">Profit Optimization via Stochastic Newsvendor Distribution Modeling</h2>
  
  <p><strong>The Challenge:</strong> Physical newspaper and magazine distribution suffered from volatile demand fluctuations across thousands of retail outlets. Over-allocation resulted in high return/waste costs, while under-allocation led to stock-outs and forfeited subscription revenue.</p>
  
  <p><strong>The Solution Architecture:</strong></p>
  <ul>
    <li>Formulated and implemented a mathematical <strong>Newsvendor stochastic optimization framework</strong> tailored to asymmetric retail costs (overage penalty vs. underage opportunity loss).</li>
    <li>Integrated historical sales velocity, day-of-week seasonality, localized demographic features, and weather variables into probabilistic demand density forecasts.</li>
    <li>Scaled distribution calculations across daily delivery schedules using automated batch pipelines on Google Cloud Platform (GCP).</li>
  </ul>

  <div class="impact-callout">
    <strong>Measurable Business Impact:</strong>
    <p>Delivered a direct <strong>16% profit margin expansion</strong> across distribution logistics, eliminating tens of thousands of unsold copies while capturing previously lost retail sales.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">Python</span>
    <span class="tech-pill">Mathematical Statistics</span>
    <span class="tech-pill">Stochastic Optimization</span>
    <span class="tech-pill">GCP</span>
    <span class="tech-pill">Pandas</span>
    <span class="tech-pill">A/B Testing</span>
  </div>
</div>

---

<div class="case-study-card">
  <div class="case-header">
    <span class="client-tag">Media &amp; Digital Subscriptions &bull; Bonnier News</span>
    <span class="role-tag">Data Scientist | Stockholm, Sweden</span>
  </div>
  <h2 class="case-title">Customer Lifetime Value (CLV) &amp; Recommendation Engine</h2>
  
  <p><strong>The Challenge:</strong> Increasing subscriber churn rates and fragmented engagement across 40+ media titles demanded targeted retention and personalized cross-selling strategies.</p>
  
  <p><strong>The Solution Architecture:</strong></p>
  <ul>
    <li>Engineered machine learning models for early churn warning, customer reactivation probability, and dynamic Customer Lifetime Value (CLV) scoring.</li>
    <li>Built a personalized recommendation engine predicting next-best magazine and digital publication purchases based on consumer reading trajectories.</li>
    <li>Conducted rigorous randomized controlled trials (A/B testing) across 40+ marketing campaigns, delivering executive-level reporting dashboards for editorial and growth stakeholders.</li>
  </ul>

  <div class="impact-callout">
    <strong>Measurable Business Impact:</strong>
    <p>Substantially boosted marketing ROI, increased digital reading engagement, and unlocked predictable subscription renewal streams.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">Python</span>
    <span class="tech-pill">Recommendation Systems</span>
    <span class="tech-pill">CLV &amp; Churn Modeling</span>
    <span class="tech-pill">GCP</span>
    <span class="tech-pill">A/B Testing</span>
  </div>
</div>

---

<div class="case-study-card">
  <div class="case-header">
    <span class="client-tag">AI Engineering &bull; Modern Stack</span>
    <span class="role-tag">ML Engineer | Independent Consulting</span>
  </div>
  <h2 class="case-title">Production RAG Architecture &amp; Stateful Multi-Agent AI (LangGraph)</h2>
  
  <p><strong>The Challenge:</strong> Enterprises require LLM applications that do not hallucinate, respect strict document permissions, and autonomously orchestrate multi-step data analysis workflows.</p>
  
  <p><strong>The Solution Architecture:</strong></p>
  <ul>
    <li>Designed end-to-end Retrieval-Augmented Generation (RAG) pipelines featuring semantic chunking, dense vector embeddings, cross-encoder reranking, and citation synthesis.</li>
    <li>Implemented stateful multi-agent systems using <strong>LangGraph</strong> and <strong>LangChain</strong> to coordinate tool execution, SQL query generation, self-correcting code checks, and human review steps.</li>
    <li>Packaged architectures into high-throughput FastAPI services with Redis-backed state caching and Docker deployment.</li>
  </ul>

  <div class="impact-callout">
    <strong>Measurable Business Impact:</strong>
    <p>Delivers deterministic, traceable AI workflows with sub-second response times and full hallucination guardrails.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">LangGraph</span>
    <span class="tech-pill">LangChain</span>
    <span class="tech-pill">Vector Databases</span>
    <span class="tech-pill">FastAPI</span>
    <span class="tech-pill">Docker</span>
    <span class="tech-pill">PyTorch</span>
  </div>
</div>

---

<div class="cta-box">
  <h2>Want to explore how these solutions apply to your systems?</h2>
  <p>I would be glad to review your current architecture and discuss implementation strategies.</p>
  <a href="/contact/" class="btn-consulting-primary">
    <i class="fas fa-comments"></i> Inquire About a Project
  </a>
</div>
