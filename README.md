# masudpervez27.github.io

Personal portfolio and consulting website of **Masud Pervez** — Senior Data Scientist & ML
Engineer, operating as an independent B2B consultant (Swedish *Enskild Firma*, F-skatt) in
Stockholm.

**Live site:** <https://masudpervez27.github.io>

The site has two jobs:

1. **Show what I have done** — employment history with measurable production outcomes.
2. **Sell what I can do for a client** — eight consulting service areas, clear engagement
   models, and a low-friction way to get in touch or request the full CV.

---

## Table of contents

- [Tech stack](#tech-stack)
- [Site map](#site-map)
- [Repository layout](#repository-layout)
- [Local development](#local-development)
- [Deployment](#deployment)
- [Editing content](#editing-content)
- [Content rules](#content-rules)
- [Design system](#design-system)
- [Security & privacy](#security--privacy)
- [Verification checklist](#verification-checklist)
- [Known issues & backlog](#known-issues--backlog)

---

## Tech stack

| Piece | Detail |
|---|---|
| Generator | Jekyll 3.10 (the version GitHub Pages runs) |
| Theme | [Minimal Mistakes](https://mmistakes.github.io/minimal-mistakes/) 4.26.2 — **vendored** into `_includes/`, `_layouts/`, `_sass/`. These files are owned by this repo and may be edited. |
| Markdown | kramdown (GFM input), Rouge highlighting |
| Styling | Theme Sass + custom overrides in `assets/css/main.scss` |
| Icons | Font Awesome 6.5.2 from cdnjs, pinned with Subresource Integrity |
| Fonts | System font stack only — no webfonts, no font CDN |
| Contact form | [FormSubmit](https://formsubmit.co/) (no backend needed) |
| Hosting | GitHub Pages, classic build from the `master` branch |
| Comments / search / analytics | All disabled |

Plugins (all GitHub-Pages-whitelisted; `plugins:` and `whitelist:` in `_config.yml` must list
the same set): `jekyll-paginate`, `jekyll-sitemap`, `jekyll-gist`, `jekyll-feed`,
`jekyll-include-cache`, `jemoji`, `jekyll-redirect-from`.

> The `github-pages` meta-gem is deliberately **not** used — it cannot resolve on Ruby ≥ 4.0.
> The `Gemfile` pins each plugin to the exact version GitHub Pages runs instead. See the
> comments in `Gemfile` for details.

---

## Site map

| Tab | URL | Source | Purpose |
|---|---|---|---|
| Home | `/` | `index.html` | Hero, key metrics, 8 service teasers, engagement teaser, CTA |
| Services | `/services/` | `_pages/services.md` | The client offer: 8 service areas (problem → deliverables → benefit) + 4 engagement models |
| Experience | `/experience/` | `_pages/experience.md` | Five past roles with what I owned and the outcome |
| About | `/about/` | `_pages/about.md` | Summary, career timeline, education, tech stack, business details, CV request (`#cv`) |
| Insights | `/posts/` | `_pages/posts.md` | Technical blog index |
| Contact | `/contact/` | `_pages/contact.md` | Email / LinkedIn / GitHub + the contact & CV-request form (`#request-cv`) |

Navigation is defined in `_data/navigation.yml`.

**Redirects** (via `jekyll-redirect-from`; a static site cannot send HTTP 301, so these are
meta-refresh pages with a canonical link):

| Old URL | New URL | Stub |
|---|---|---|
| `/projects/` | `/experience/` | `_pages/projects-redirect.md` |
| `/cv/` | `/about/#cv` | `_pages/cv-redirect.md` |

---

## Repository layout

```
.
├── _config.yml              # Site settings, author sidebar, plugins, defaults
├── Gemfile                  # Pinned gems matching GitHub Pages
├── index.html               # Homepage (pure HTML — NOT processed as markdown)
├── _pages/                  # All standalone pages (each needs an explicit permalink)
├── _posts/                  # Blog posts: YYYY-MM-DD-slug.md
├── _data/
│   ├── navigation.yml       # Top navigation (6 items)
│   └── ui-text.yml          # Theme UI strings
├── _includes/               # Vendored theme partials + custom ones:
│   ├── author-profile-custom-links.html  # Sidebar "Available" badge + CV button
│   ├── footer.html                       # Custom footer
│   └── head/custom.html                  # Favicon
├── _layouts/                # Vendored theme layouts
├── _sass/                   # Vendored theme Sass
├── assets/
│   ├── css/main.scss        # Theme import + ALL custom component styles
│   ├── images/              # Avatar, monogram, header images
│   └── js/                  # Theme JS (main.min.js)
├── AGENT.md                 # Maintainer/AI-agent reference (excluded from build)
└── .claude/CLAUDE.md        # Same reference for Claude Code (keep in sync with AGENT.md)
```

---

## Local development

### Option A — Docker (no Ruby install needed)

```powershell
docker run --rm -it -p 4000:4000 -v "${PWD}:/srv/jekyll" -w /srv/jekyll ruby:3.3 `
  bash -c "bundle install && bundle exec jekyll serve --host 0.0.0.0 --livereload"
```

Open <http://localhost:4000>.

### Option B — Native Ruby

Requires Ruby ≥ 3.0 (tested up to 4.0) and Bundler. On Windows, use RubyInstaller **with
the MSYS2 devkit** (native gems need a compiler).

```bash
bundle install
bundle exec jekyll serve          # http://localhost:4000, rebuilds on change
bundle exec jekyll build --trace  # one-off build into _site/
```

Notes:

- `_config.yml` is **not** reloaded by `jekyll serve` — restart after editing it.
- Version traps already handled in `Gemfile`: `liquid` must be ≥ 4.0.4 on Ruby ≥ 3.2, and
  Ruby ≥ 3.4 "bundled gems" (`csv`, `base64`, `bigdecimal`, …) must be declared.
- `Gemfile.lock`, `_site/`, `.jekyll-cache/` are git-ignored.

---

## Deployment

Push to `master`. GitHub Pages builds and publishes automatically (usually 1–3 minutes).
There is no CI workflow. If a change is not visible yet, wait and hard-refresh before
assuming the build failed; build errors are shown under **Settings → Pages** and in the
repository's **Actions** tab.

---

## Editing content

### Add a blog post

Create `_posts/YYYY-MM-DD-slug.md`:

```yaml
---
title: "Post title"
date: 2026-10-21T10:00:00+02:00
categories: [Machine Learning]
tags: [Python, MLOps]
header:
  teaser: /assets/images/bg-post-resize.jpg
toc: true
toc_label: "Contents"
---
```

`layout`, `author_profile`, `read_time`, `share`, and `related` come from `_config.yml`
defaults. The post is published at `/posts/<slug>/`.

### Add or change a service

1. Edit the card in `_pages/services.md` (copy an existing `<div class="service-card" id="…">`
   block). The card title is an `<h2 class="service-title">`.
2. Update the matching teaser card on `index.html` (same title, one-line description,
   `Learn more` link to `/services/#<id>`).
3. Keep the anchor id stable — the homepage links to it.

### Add or change a role

Edit `_pages/experience.md` (one `case-study-card` per employer) **and** the one-line entry
in the timeline on `_pages/about.md`.

### Gotcha: markdown inside HTML

kramdown does **not** process markdown inside HTML blocks (`<div>…</div>`), and `index.html`
is never processed as markdown at all. Inside those, write HTML (`<h2>`, `<ul><li>`, `<hr>`,
`<strong>`), not `##`, `*`, `---`, or `**`.

---

## Content rules

1. **Employment ≠ consulting.** `/experience/` lists past *employers* only and never calls
   them clients. Consulting capabilities live on `/services/`. Service cards may name an
   employer as plain text ("Built in production at SEB") but never link to `/experience/`.
2. **One fact, one place.** Metrics appear in the metric cards and the relevant employer
   card; other pages link rather than repeat.
3. **Generic service titles.** Named techniques (Newsvendor, XGBoost, …) belong in
   `/experience/` or blog posts, not service titles.
4. **Consistent numbers.** Experience is stated as "10+ years" / "ten years" everywhere.
5. **Anchors that must exist exactly once:** `#request-cv` (contact form), `#cv` (About),
   `#engagement-levels` and the eight service ids (Services).

---

## Design system

All custom styling lives in `assets/css/main.scss` and uses CSS variables defined in
`:root` (`--primary-accent`, `--dark-slate`, `--card-border`, …).

| Component | Class | Used on |
|---|---|---|
| Dark hero | `.hero-consulting` | Home |
| Page tagline pill | `.hero-tagline` | About, Services, Experience, Contact |
| Metric cards | `.metrics-grid` > `.metric-card` | Home |
| Service cards | `.services-grid` > `.service-card` | Home, About, Contact, Services |
| Employer cards | `.case-study-card` | Experience |
| Green info banner | `.b2b-engagement-banner` | Home, Services, About, Contact |
| Dark call-to-action | `.cta-box` | Most pages |
| Buttons | `.btn-consulting-primary` / `.btn-consulting-secondary` | Everywhere (secondary is white-on-dark — use only on dark backgrounds) |
| Career timeline | `.timeline-list` | About |

Several utility classes (`.service-features`, `.service-icon-box`, `.impact-callout`) only
work **inside** their parent card. A Sass compile error breaks styling for the whole site —
build locally before pushing CSS changes.

---

## Security & privacy

| Area | Status |
|---|---|
| Third-party CSS | Font Awesome is loaded with an SRI `integrity` hash + `crossorigin`. When upgrading Font Awesome, update the hash from `https://api.cdnjs.com/libraries/font-awesome/<version>?fields=sri`. |
| External links | `target="_blank"` links use `rel="noopener noreferrer"`. |
| Contact form | FormSubmit with its captcha enabled and a `_honey` honeypot field; input lengths are capped. |
| Email exposure | The email address is public (sidebar, mailto links, form action). Expect some spam. |
| Secrets | None. The site is fully static; there are no API keys in the repo. Never commit any. |
| Redirect map | `redirect_from.json: false` stops publication of `/redirects.json`. |
| Build scope | `README.md`, `AGENT.md`, `CLAUDE.md`, `Gemfile*` are excluded from the published site. |
| Tracking | No analytics, no cookies, no comments provider. |

**Recommended follow-ups** (require action in external accounts):

- **FormSubmit alias:** after the first submission, FormSubmit emails an activation link and
  offers a random alias (e.g. `https://formsubmit.co/el/xxxxxx`). Replace the plain email in
  the form `action` with that alias so the address is not harvestable from the HTML.
- **Privacy notice:** form submissions are processed by FormSubmit (a third party). For full
  GDPR transparency, consider a short `/privacy/` page naming the processor and retention.
- **Enforce HTTPS** in *Settings → Pages* (on by default for `github.io`).
- **Custom domain** (optional, recommended for a consultancy) — add a `CNAME` file and DNS
  records, then update `url` in `_config.yml`.

---

## Verification checklist

After `bundle exec jekyll build`:

```bash
grep -c 'id="request-cv"' _site/contact/index.html            # 1
grep -c 'id="cv"' _site/about/index.html                       # 1
grep -o 'masthead__menu-item' _site/index.html | wc -l         # 6
grep -c 'class="service-card"' _site/services/index.html       # 8
grep -c 'case-study-card' _site/experience/index.html          # 5
grep -o 'http-equiv="refresh"[^>]*' _site/projects/index.html  # → /experience/
grep -o 'http-equiv="refresh"[^>]*' _site/cv/index.html        # → /about/#cv
grep -n '## \|^\* \*\*' _site/index.html _site/about/index.html  # nothing (no raw markdown)
ls _site/redirects.json _site/README.md                        # both must NOT exist
```

Live:

```bash
curl -sI https://masudpervez27.github.io/experience/   # 200
curl -so /dev/null -w '%{http_code} %{size_download}\n' \
     https://masudpervez27.github.io/assets/css/main.css  # 200, non-trivial size
```

---

## Known issues & backlog

- **Unused images (~4 MB):** `bg-contact.jpg`, `bg-contact-resize2.png`, `bg-index.jpg`,
  `bg-index-resize.jpg`, `bg-post.jpg`, `foo-bar-identity-th.jpg`, `headshot-2.jpeg`,
  `headshot-3.jpeg`, `Intent-Classification-with-BERT.png`, `unsplash-image-8.jpg` are not
  referenced anywhere. Delete them or move them out of `assets/` (they are still published).
- **Avatar size:** `bio-photo.jpeg` is ~580 KB for a ~150 px avatar. Resize/compress to
  < 60 KB and strip EXIF metadata (location data) before publishing photos.
- **Open Graph image:** `og_image` points to the avatar; a dedicated 1200×630 social card
  would look better on LinkedIn shares.
- **Unused `projects` collection** is still declared in `_config.yml` (harmless).
- **Service overlap:** "Data-Driven Decision Making" and "Forecasting" both mention demand /
  inventory optimisation — consider keeping it in one place.
- **Fact check:** Service 6 says GCP was used at SEB, while the SEB role lists Hadoop /
  on-prem tooling. Confirm and align.
- **Contact email:** a domain address (e.g. `masud@yourdomain.se`) looks more professional
  than a Yahoo address for B2B clients.

---

© Masud Pervez. Theme: Minimal Mistakes by Michael Rose (MIT).