---
title: "Contact & Consultation"
layout: single
permalink: /contact/
author_profile: true
classes: wide
---

<div class="hero-tagline" style="margin-bottom: 1.5rem;">
  <i class="fas fa-handshake"></i><span>Direct B2B Inquiries &bull; Stockholm, Sweden</span>
</div>

Whether you are looking to bring in a senior specialist for an upcoming machine learning initiative, audit an existing fraud prevention architecture, or request a complete Curriculum Vitae (PDF), I welcome the opportunity to connect.

---

<div class="services-grid" style="margin-top: 1.5rem; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));">
  <div class="service-card">
    <div class="service-icon-box">
      <i class="fas fa-envelope-open-text"></i>
    </div>
    <h3 class="service-title">Direct Email</h3>
    <p class="service-desc">
      Send project briefs, RFPs, or CV requests directly. Typically answered within 24 hours.
    </p>
    <p>
      <a href="mailto:masud.pervez@yahoo.com?subject=Consulting%20Inquiry%20/%20CV%20Request" class="btn-consulting-primary" style="font-size: 0.88rem; padding: 0.6rem 1rem;"><i class="fas fa-paper-plane"></i><span>Send Email</span></a>
    </p>
  </div>

  <div class="service-card">
    <div class="service-icon-box">
      <i class="fab fa-linkedin-in"></i>
    </div>
    <h3 class="service-title">LinkedIn Professional</h3>
    <p class="service-desc">
      Connect on LinkedIn for professional networking, recommendations, and direct messaging.
    </p>
    <p>
      <a href="https://www.linkedin.com/in/masud-pervez-431a0418" target="_blank" rel="noopener noreferrer" class="btn-consulting-primary" style="font-size: 0.88rem; padding: 0.6rem 1rem;"><i class="fab fa-linkedin"></i><span>Connect on LinkedIn</span></a>
    </p>
  </div>

  <div class="service-card">
    <div class="service-icon-box">
      <i class="fab fa-github"></i>
    </div>
    <h3 class="service-title">GitHub Portfolio</h3>
    <p class="service-desc">
      Explore code repositories, open-source implementations, and machine learning architectures.
    </p>
    <p>
      <a href="https://github.com/masudpervez27" target="_blank" rel="noopener noreferrer" class="btn-consulting-primary" style="font-size: 0.88rem; padding: 0.6rem 1rem;"><i class="fab fa-github"></i><span>View GitHub Profile</span></a>
    </p>
  </div>
</div>

---

<div id="request-cv" class="consulting-contact-form-card">
  <h2 style="margin-top: 0; color: var(--dark-slate); font-weight: 800;"><i class="fas fa-envelope"></i><span>Send a Message or Request Full CV</span></h2>
  <p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem;">
    Fill out the form below to initiate an inquiry or request my full CV (PDF) with detailed role descriptions and professional references. Messages are delivered directly to my inbox.
  </p>

  <form action="{{ site.contact_form_action }}" method="POST">
    <!-- FormSubmit Configuration -->
    <input type="hidden" name="_subject" value="New Inquiry / CV Request via masudpervez27.github.io">
    <input type="hidden" name="_captcha" value="true">
    <input type="hidden" name="_template" value="table">
    <input type="hidden" name="_next" value="{{ '/contact/thanks/' | absolute_url }}">
    <!-- honeypot: bots fill it, humans never see it -->
    <input type="text" name="_honey" style="display:none" tabindex="-1" autocomplete="off" aria-hidden="true">

    <div class="form-row">
      <div class="form-group">
        <label for="name">Your Name *</label>
        <input type="text" id="name" name="name" required maxlength="120" autocomplete="name" placeholder="e.g. Johan Andersson">
      </div>
      <div class="form-group">
        <label for="email">Work / Contact Email *</label>
        <input type="email" id="email" name="email" required maxlength="254" autocomplete="email" placeholder="name@company.com">
      </div>
    </div>

    <div class="form-row">
      <div class="form-group">
        <label for="company">Company / Organization</label>
        <input type="text" id="company" name="company" maxlength="160" autocomplete="organization" placeholder="e.g. Financial Services AB">
      </div>
      <div class="form-group">
        <label for="inquiry_type">Inquiry Type *</label>
        <select id="inquiry_type" name="inquiry_type" required>
          <option value="General consultation" selected>General consultation</option>
          <option value="Project inquiry (diagnostic, proof of concept or build)">Project inquiry (diagnostic, proof of concept or build)</option>
          <option value="Contract or interim role">Contract or interim role</option>
          <option value="Request full CV (PDF)">Request full CV (PDF)</option>
          <option value="Other">Other</option>
        </select>
      </div>
    </div>

    <div class="form-group">
      <label for="message">Project Overview / Request Details *</label>
      <textarea id="message" name="message" rows="5" required maxlength="5000" placeholder="Please outline your project timeline, requirements, or CV request details..."></textarea>
    </div>

    <p style="font-size: 0.8rem; color: #64748b; margin: 0 0 1rem 0;">
      Submissions are relayed by <a href="https://formsubmit.co/" target="_blank" rel="noopener noreferrer">FormSubmit</a> to my inbox and used only to reply to your inquiry.
    </p>

    <button type="submit" class="btn-consulting-primary" style="border: none; padding: 0.85rem 1.75rem; font-size: 1rem; width: 100%; justify-content: center;">
      <i class="fas fa-paper-plane"></i><span>Submit Inquiry</span>
    </button>
  </form>

  <div style="margin-top: 1.25rem; font-size: 0.82rem; color: #64748b; text-align: center;">
    Prefer writing directly from your email client? Send an email to <a href="mailto:masud.pervez@yahoo.com?subject=Consulting%20Inquiry%20/%20CV%20Request">masud.pervez@yahoo.com</a>.
  </div>
</div>

---

<div class="b2b-engagement-banner">
  <h3><i class="fas fa-info-circle"></i><span>Business &amp; Administrative Details</span></h3>
  <div class="engagement-perks">
    <div class="perk-item">
      <i class="fas fa-map-pin"></i>
      <div>
        <strong>Consulting Base</strong>
        <span>Stockholm, Sweden (CET / CEST timezone).</span>
      </div>
    </div>
    <div class="perk-item">
      <i class="fas fa-file-invoice"></i>
      <div>
        <strong>Corporate Invoicing</strong>
        <span>Swedish Enskild Firma with approved F-tax (F-skatt sedel). Direct invoice with standard 30-day payment terms.</span>
      </div>
    </div>
    <div class="perk-item">
      <i class="fas fa-globe-europe"></i>
      <div>
        <strong>Working Modality</strong>
        <span>On-site in Stockholm, hybrid, or fully remote across Sweden, the EU, and international markets.</span>
      </div>
    </div>
    <div class="perk-item">
      <i class="fas fa-shield-alt"></i>
      <div>
        <strong>Privacy &amp; GDPR</strong>
        <span>Your details are used only to respond to your inquiry, are not sold or used for marketing, and are deleted on request.</span>
      </div>
    </div>
  </div>
</div>
