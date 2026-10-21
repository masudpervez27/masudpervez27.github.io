---
title: "Professional Experience"
layout: single
permalink: /experience/
author_profile: true
classes: wide
description: "Senior Data Scientist and Machine Learning Engineer roles across banking, media publishing, eye-tracking analytics, clinical research, and retail supply chain — as an employee, with measurable production outcomes."
---

<div class="hero-tagline" style="margin-bottom: 1.5rem;">
  <i class="fas fa-briefcase"></i><span>Five Organisations &bull; Banking &bull; Media &bull; HCI &bull; Clinical Research &bull; Retail</span>
</div>

These are roles I held as an **employee** (and, at Karolinska Institutet, as a master's-thesis
statistician) — positions I held and systems I owned in production, not client engagements. I am
now bringing that same ownership to consulting work.

Every entry below covers what I was responsible for, the technology I ran, and the measurable
result it produced.

---

<div class="case-study-card" id="seb">
  <div class="case-header">
    <span class="client-tag">Banking &amp; Finance</span>
    <span class="role-tag">Senior Data Scientist &bull; 2021 – Present &bull; Stockholm, Sweden</span>
  </div>
  <h2 class="case-title">Skandinaviska Enskilda Banken (SEB)</h2>

  <p><strong>What I owned:</strong></p>
  <ul>
    <li>Architecture and live deployment of real-time machine learning fraud detection, delivered as microservices with low-latency scoring endpoints.</li>
    <li>A cost-based threshold optimization framework that prices false positives against false negatives using the bank's own loss structure, so the business can trade fraud loss against customer friction explicitly.</li>
    <li>Modular Apache Spark pipelines on the on-premise Hadoop platform for distributed feature computation, batch inference, and metric logging across high-volume transactional data.</li>
    <li>The move of model workloads from on-premise Hadoop to Google Cloud Platform: Terraform for the model training infrastructure and Dataform for the feature engineering pipelines.</li>
    <li>End-to-end model governance — concept drift detection, distribution-shift monitoring, and automated recalibration and retraining protocols.</li>
    <li>Technical due diligence on external fraud-prevention vendors, and cross-functional work with engineering, compliance, and product on financial-crime defences.</li>
  </ul>

  <div class="impact-callout">
    <strong>Outcome:</strong>
    <p>Over <strong>60–70% fraud detection recall</strong> in live production while holding false positives low enough that legitimate customer payments were not interrupted.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">Python</span>
    <span class="tech-pill">Apache Spark</span>
    <span class="tech-pill">Scala</span>
    <span class="tech-pill">Hadoop / HDFS</span>
    <span class="tech-pill">Google Cloud Platform</span>
    <span class="tech-pill">Terraform</span>
    <span class="tech-pill">Dataform</span>
    <span class="tech-pill">Scikit-Learn</span>
    <span class="tech-pill">Microservices</span>
    <span class="tech-pill">Model Governance</span>
  </div>
</div>

---

<div class="case-study-card" id="bonnier-news">
  <div class="case-header">
    <span class="client-tag">Media &amp; Publishing</span>
    <span class="role-tag">Data Scientist &bull; 2018 – 2021 &bull; Stockholm, Sweden</span>
  </div>
  <h2 class="case-title">Bonnier News</h2>

  <p><strong>What I owned:</strong></p>
  <ul>
    <li>A stochastic Newsvendor optimization model for physical publication distribution, balancing overstock waste against lost sales under asymmetric costs.</li>
    <li>Predictive models for churn warning, subscription reactivation, and dynamic Customer Lifetime Value scoring.</li>
    <li>A personalized recommendation system predicting next-best magazine and digital purchase from reading trajectories.</li>
    <li>Demand density forecasts built from sales velocity, day-of-week seasonality, localized demographics, and weather variables, scaled through automated GCP batch pipelines.</li>
    <li>Rigorous A/B experimentation and retention analytics across 40+ publications, with executive dashboards for growth and editorial stakeholders.</li>
  </ul>

  <div class="impact-callout">
    <strong>Outcome:</strong>
    <p>A direct <strong>16% profit margin expansion</strong> in distribution logistics, alongside materially improved marketing ROI and digital reading engagement.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">Python</span>
    <span class="tech-pill">Mathematical Statistics</span>
    <span class="tech-pill">Stochastic Optimization</span>
    <span class="tech-pill">Google Cloud Platform</span>
    <span class="tech-pill">Pandas</span>
    <span class="tech-pill">A/B Testing</span>
  </div>
</div>

---

<div class="case-study-card" id="tobii-pro">
  <div class="case-header">
    <span class="client-tag">Human-Computer Interaction</span>
    <span class="role-tag">Data Scientist &amp; Statistician &bull; 2017 – 2018 &bull; Stockholm, Sweden</span>
  </div>
  <h2 class="case-title">Tobii Pro</h2>

  <p><strong>What I owned:</strong></p>
  <ul>
    <li>Preprocessing and analysis of high-frequency eye-tracking survey datasets for consumer attention and behavioural pattern discovery.</li>
    <li>Statistical modelling with principal component analysis, logistic regression, and decision trees to reduce dimensionality without losing signal.</li>
    <li>Interactive stakeholder dashboards that made eye-tracking findings usable by product and research teams, not just analysts.</li>
  </ul>

  <div class="impact-callout">
    <strong>Outcome:</strong>
    <p>Turned raw high-volume gaze data into interpretable behavioural insight that non-technical stakeholders could act on directly.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">Python</span>
    <span class="tech-pill">R</span>
    <span class="tech-pill">PCA</span>
    <span class="tech-pill">Regression</span>
    <span class="tech-pill">Dashboards</span>
  </div>
</div>

---

<div class="case-study-card" id="karolinska-institutet">
  <div class="case-header">
    <span class="client-tag">Clinical Research</span>
    <span class="role-tag">Statistician (Master's Project) &bull; 2012 – 2013 &bull; Stockholm, Sweden</span>
  </div>
  <h2 class="case-title">Clinical Cancer Epidemiology — Karolinska Institutet</h2>

  <p><strong>What I owned:</strong></p>
  <ul>
    <li>Epidemiological survey analytics for gynecological cancer survivors, on a multi-year clinical dataset.</li>
    <li>Factor analysis to extract latent symptom constructs from patient-reported outcomes.</li>
    <li>Hierarchical cluster comparison and validation techniques to test whether symptom groupings held up under scrutiny.</li>
  </ul>

  <div class="impact-callout">
    <strong>Outcome:</strong>
    <p>Produced validated symptom structures that supported the clinical research team's conclusions on survivorship patterns.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">R</span>
    <span class="tech-pill">Factor Analysis</span>
    <span class="tech-pill">Clustering</span>
    <span class="tech-pill">Survey Analytics</span>
  </div>
</div>

---

<div class="case-study-card" id="marks-and-spencer">
  <div class="case-header">
    <span class="client-tag">Retail &amp; Supply Chain</span>
    <span class="role-tag">Planner &amp; Data Integrity Analyst &bull; 2007 – 2008 &bull; Dhaka, Bangladesh</span>
  </div>
  <h2 class="case-title">Marks &amp; Spencer</h2>

  <p><strong>What I owned:</strong></p>
  <ul>
    <li>International supply chain logistics analysis, tracking how product moved through the distribution network.</li>
    <li>Shipment integrity KPIs — owning the data quality and measurement behind delivery performance reporting.</li>
    <li>Supplier performance training, turning analytical findings into better practice with third-party suppliers.</li>
  </ul>

  <div class="impact-callout">
    <strong>Outcome:</strong>
    <p>Established reliable shipment-integrity reporting and lifted supplier data discipline across the network.</p>
  </div>

  <div class="tech-pills">
    <span class="tech-pill">Supply Chain Analytics</span>
    <span class="tech-pill">KPI Reporting</span>
    <span class="tech-pill">Data Integrity</span>
  </div>
</div>

---

<div class="cta-box">
  <h2>Ready to put this experience to work on your systems?</h2>
  <p>I take on consulting engagements with companies of all sizes — from growing SMEs to large enterprises. Fixed-scope projects, embedded contractor roles, and strategic advisory welcome.</p>
  <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
    <a href="/services/" class="btn-consulting-primary"><i class="fas fa-briefcase"></i><span>Explore My Services</span></a>
    <a href="/contact/" class="btn-consulting-secondary"><i class="fas fa-envelope"></i><span>Get in Touch</span></a>
  </div>
</div>
