# AGENT.md — masudpervez27.github.io

Personal data-science & AI-consulting portfolio. Jekyll site on the `minimal-mistakes` theme,
deployed to **https://masudpervez27.github.io** via GitHub Pages (classic, `master` branch).

---

## Stack

| Piece | Detail |
|---|---|
| Generator | Jekyll 3.10 (via `github-pages` gem) |
| Theme | `remote_theme: mmistakes/minimal-mistakes@4.26.2` — **vendored locally** in `_includes/`, `_layouts/`, `_sass/`. These are owned files, not a submodule. |
| Sass | `_sass/minimal-mistakes/` + overrides in `assets/css/main.scss` |
| Icons | Font Awesome **6.5.2** via cdnjs (`_includes/head.html:20`). FA6-only icons (`fa-file-contract`, `fa-chart-line`, `fa-tachometer-alt`) are safe. |
| Fonts | System font stacks only. No webfonts. |
| Comments | Disabled (`site.comments.provider` is blank — comments never render) |
| Search | Disabled (`site.search` is blank) |

### Plugins (all GitHub-Pages-whitelisted)

`_config.yml` → `plugins:` and `whitelist:` must list the **same** set:

```
jekyll-paginate, jekyll-sitemap, jekyll-gist, jekyll-feed,
jekyll-include-cache, jemoji, jekyll-redirect-from
```

`jekyll-redirect-from` is **not** in `DEFAULT_PLUGINS` — it must be listed explicitly or the
`/projects/` and `/cv/` redirects will not be generated.

`redirect_from: { json: false }` in `_config.yml` suppresses the public `/redirects.json`
that the plugin would otherwise publish.

### The `github-pages` meta-gem is NOT used

`Gemfile` pins the individual plugins to the exact versions GitHub Pages itself runs
(github-pages 232 → jekyll 3.10.0, liquid 4.0.4, kramdown 2.4.0,
jekyll-redirect-from 0.16.0, jekyll-include-cache 0.2.1, jekyll-feed 0.17.0,
jekyll-sitemap 1.4.0, jekyll-gist 1.5.0, jekyll-paginate 1.1.0, jemoji 0.13.0).

**Do not reintroduce the `github-pages` meta-gem.** It transitively requires
`jekyll-commonmark-ghpages` → `commonmarker ~> 0.22`, whose gemspec caps Ruby at `< 4.0`,
so it cannot resolve on a modern local Ruby. This site uses kramdown
(`markdown: kramdown`), never commonmark, and the theme is vendored locally — the
individual plugins are a complete substitute.

Two more version traps, both of which break the build on a modern Ruby:

- **liquid must be ≥ 4.0.4.** Liquid 4.0.3 calls `Object#tainted?`, removed in Ruby 3.2 →
  `undefined method 'tainted?' for false`.
- **Declare the "bundled gems" in the Gemfile.** Ruby 3.4 moved `csv`, `base64`,
  `bigdecimal`, `mutex_m`, `observer`, `abbrev`, `nkf`, `drb`, `getoptlong`, and `ostruct`
  out of "default gems" into "bundled gems". Bundler hides bundled gems unless they are
  declared, so Jekyll fails to load without them. All are pure-Ruby, so they are safe for
  GitHub Pages' own build environment (which ignores the Gemfile entirely).

---

## Site map (6 nav tabs)

| Page | Path | Source | Responsibility |
|---|---|---|---|
| Home | `/` | `index.html` | Hero, metric cards, 8 slim pillar cards linking to `/services/#…`, engagement teaser, CTA |
| Services | `/services/` | `_pages/services.md` | **The offer to the client.** 8 service areas + engagement models |
| Experience | `/experience/` | `_pages/experience.md` | **Employers only** — 5 real employee roles with outcomes |
| About | `/about/` | `_pages/about.md` | Executive summary, condensed career timeline, education, tech arsenal, **CV section (`id="cv"`)** |
| Insights | `/posts/` | `_pages/posts.md` | Blog index (`layout: collection`) |
| Contact | `/contact/` | `_pages/contact.md` | Contact channels + **the one and only form** (`id="request-cv"`) |

### Redirects

| Old URL | Redirects to | Mechanism |
|---|---|---|
| `/projects/` | `/experience/` | `_pages/projects-redirect.md` → `redirect_to` |
| `/cv/` | `/about/#cv` | `_pages/cv-redirect.md` → `redirect_to` |

The redirect is **not** an HTTP 301 — a static Pages site cannot emit one. The plugin
renders a 200 page containing `<meta http-equiv="refresh" content="0; url=…">`,
`<script>location=…</script>`, `<link rel="canonical">`, and `<meta name="robots" content="noindex">`.
The browser treats `content="0"` as an immediate redirect; the canonical tag carries the SEO signal.

The plugin's `generator.rb` does `doc.content = doc.output = page.output`, so the stub's
own body is **discarded** whenever the plugin loads. A one-line "this page has moved" body
is still worth keeping: it is the fail-safe if the plugin ever fails to load, in which case
the stub renders as a normal page instead of a blank one. `sitemap: false` keeps the stubs
out of the sitemap (the plugin forces it anyway — declaring it is explicit hygiene).

The plugin ships its own `redirect.html` layout and injects it as `site.layouts["redirect"]`.
**Do not author a `_layouts/redirect.html`** — it would shadow the plugin's.

---

## Content rules

### 1. Employment vs consulting — keep this distinction honest

Masud's SEB / Bonnier News / Tobii Pro / Karolinska / M&S roles were **employment**, not client
engagements. The consultancy is new. Therefore:

- `/experience/` lists **employers only**, framed as *"roles I held and systems I owned in
  production"*. Never call them client projects or engagements.
- The RAG / LangGraph work is **not** employment — it is a capability and lives on `/services/`.
- Service cards may name an employer as **plain text only** ("Built in production at SEB") —
  **never as a hyperlink** to `/experience/`.
- `/services/` speaks to a client who has not worked with Masud before: problem → deliverable →
  the client's benefit.

### 2. No duplicate content across pages

Each fact lives in exactly one place. Metrics (60–70% recall, +16% margin) appear in the metric
cards and in the relevant employer card, but the homepage does **not** restate service
deliverables in full, and `/about/` does not repeat the employer detail that `/experience/`
holds.

### 3. Generic service titles

Service area names stay generic — "Real-Time ML Systems & Intelligent Automation", not "Fraud
Detection Model". Named techniques and model types (Newsvendor, XGBoost, LightGBM) belong in
`/experience/` or a blog post, not in a service title.

### 4. Years-of-experience consistency

The site says **10+ years**. `_pages/about.md` previously said "eight years" — a fixed
inconsistency. Keep all mentions at "10+ years" / "ten years".

---

## Anchor contract

| Anchor | Defined in | Consumed by |
|---|---|---|
| `id="request-cv"` | `_pages/contact.md` (the form card) | 5 inbound links + sidebar button. **Must exist exactly once site-wide.** |
| `id="cv"` | `_pages/about.md` (CV section wrapper div) | `/about/#cv` — the redirect target for `/cv/` |
| `id="engagement-levels"` | `_pages/services.md` | Homepage "See Engagement Models" button |
| `id="<service-slug>"` | `_pages/services.md` | Homepage "Learn more →" links |

The CV section uses an **explicit `<div id="cv">`**, not a markdown heading — kramdown
`auto_ids` would slugify the heading text to `curriculum-vitae-credentials`, not `cv`.

Service-card anchors: `real-time-ml-systems`, `production-mlops`,
`enterprise-generative-ai`, `data-driven-decision-making`,
`forecasting-predictive-analytics`, `cloud-data-ai-platform`, `business-dashboards`,
`ai-automation-agentic-workflow`.

Employer anchors on `/experience/`: `seb`, `bonnier-news`, `tobii-pro`,
`karolinska-institutet`, `marks-and-spencer`.

---

## SCSS structure & gotchas

`assets/css/main.scss` holds theme overrides. **Nesting matters** — several utility classes only
exist inside a parent selector:

| Class | Where it lives | Consequence |
|---|---|---|
| `.service-features`, `.tech-pills` | nested inside `.service-card` | A **top-level** `.tech-pills` rule was added so employer cards on `/experience/` can use pills. New top-level usage needs its own rule. |
| `.impact-callout` | nested inside `.case-study-card` | Fine for employer cards (they use `.case-study-card`). |
| `.service-icon-box`, `.service-title`, `.service-desc` | nested inside `.service-card` | Same — reuse the parent card. |

New top-level rules added: `.tech-pills`, `.service-subtitle`, `.service-benefit`,
`.service-provenance`, `.service-link`, `.timeline-list` (+ `.timeline-item`,
`.timeline-period`, `.timeline-role`, `.timeline-org`), `.cv-section`.

**A SCSS compile error breaks styling for the entire site.** Keep CSS changes in their own
commit and verify `main.css` serves correctly after pushing.

---

## Build & deploy

### Local build (Windows)

Ruby is installed per-user at `C:\Users\user\rubyportable` (no admin needed). It is
**Ruby 4.0.7**, and the MSYS2 UCRT64 devkit provides the compiler for the native gems
(`racc`, `json`, `commonmarker`, `http_parser.rb`, `eventmachine`).

```bash
export PATH="/c/Users/user/rubyportable/bin:/c/Users/user/rubyportable/msys64/ucrt64/bin:/c/Users/user/rubyportable/msys64/usr/bin:$PATH"
export MSYSTEM=UCRT64
cd "D:/masudpervez27.github.io"
bundle install
bundle exec jekyll build --trace
```

The devkit PATH is required — without `ucrt64/bin` on PATH the native gems fail to build.
If the devkit is missing: `cmd //c "C:\Users\user\rubyportable\bin\ridk.cmd install 1 3"`
(the `cmd //c` form is required — the plain `cmd /c` form silently no-ops), then
`ridk enable` to put `ucrt64/bin` on PATH.

`Gemfile` must keep its `source "https://rubygems.org"` line — without it bundler has no remote
to resolve from and fails with "Could not find gem 'jekyll'".

**Offline installs:** the Ruby installation ships `.gem` files for the bundled gems in
`C:\Users\user\rubyportable\lib\ruby\gems\4.0.0\cache`. If the network is unavailable,
resolve from disk with
`BUNDLE_CACHE_PATH="/c/Users/user/rubyportable/lib/ruby/gems/4.0.0/cache" bundle install --local`.

**Cache note (verified, not a gotcha):** `_layouts/default.html` uses
`{% include_cached masthead.html %}`, and `jekyll-include-cache`'s cache key is the include
path plus explicit params only — `site.data` is **not** part of it, so a stale nav would in
principle be possible. In practice it is not, and the reason is worth knowing:

- Jekyll 3.10 has **no `Jekyll::Cache` class**, so the plugin falls back to its in-memory
  `JekyllIncludeCache::Cache`, and it registers a `site, :pre_render` hook that **clears the
  cache on every build**. No `.jekyll-cache` directory is created and no disk cache is used.
- The cache is also per-process, so a production Pages build always starts cold.

Empirically confirmed: adding a 7th nav item and rebuilding produced 7 nav items in the
output with no cache clearing. `rm -rf .jekyll-cache` before building is still harmless
insurance if you ever upgrade to a `jekyll-include-cache` version that persists to disk
(0.3.x deliberately stops using `Jekyll::Cache` and keeps the in-memory behaviour).

### Deploy

Push to `master`. GitHub Pages rebuilds automatically (~1–3 min). There is no CI workflow and no
`gh` CLI in this environment — poll the live site with `curl` to confirm.

---

## Branding

| Element | Source |
|---|---|
| Monogram mark | `assets/images/mp-monogram.svg` — an inline "MP" vector, no binary asset |
| Masthead logo | `site.logo: "/assets/images/mp-monogram.svg"` |
| Masthead name | `site.masthead_title: "Masud Pervez"` — the name alone; the pipe + "AI & Data Consulting" was doing the subtitle's job |
| Masthead tagline | `site.subtitle: "Data Science · MLOps · AI Consulting"` — a three-pillar service line, sentence case, not ALL CAPS teal |
| Favicon | `_includes/head/custom.html` — `rel="icon"` + `apple-touch-icon` pointing at the monogram |

`site.masthead_title` and `site.subtitle` are referenced **only** in `_includes/masthead.html`
— they do not affect `<title>` tags, SEO, or the footer. `site.title` ("Masud Pervez") and
`site.description` are separate and unchanged.

The minimal-mistakes default `assets/images/apple-touch-icon.png` ("MM" monogram) was
**deleted** — it was off-brand and it was the iOS home-screen icon.

---

## Verification

### Static (no build needed)

```bash
# front matter parses on every page and post
python /tmp/fmcheck.py

# nav has exactly 6 items
grep -c "title:" _data/navigation.yml

# no stale links remain
grep -rn 'href="/projects/\|href="/cv/"' --include="*.md" --include="*.html" . | grep -v _sass
```

### Local build output

```bash
ls _site/experience/index.html _site/services/index.html _site/about/index.html
grep -o 'http-equiv="refresh"[^>]*' _site/projects/index.html   # → /experience/
grep -o 'http-equiv="refresh"[^>]*' _site/cv/index.html         # → /about/#cv
grep -c 'id="request-cv"' _site/contact/index.html               # → 1
grep -c 'id="cv"' _site/about/index.html                         # → 1
grep -o 'masthead__menu-item' _site/index.html | wc -l           # → 6
ls _site/redirects.json                                          # → must NOT exist
grep -c 'class="service-card"' _site/services/index.html         # → 8
grep -c 'case-study-card' _site/experience/index.html            # → 5
```

### Live

```bash
curl -sI https://masudpervez27.github.io/experience/             # 200
curl -s  https://masudpervez27.github.io/projects/ | grep -o 'url=[^"]*'
curl -s  https://masudpervez27.github.io/cv/       | grep -o 'url=[^"]*'
curl -s  https://masudpervez27.github.io/          | grep -c 'masthead__menu-item'   # → 6
curl -so /dev/null -w '%{http_code} %{size_download}\n' \
     https://masudpervez27.github.io/assets/css/main.css          # → 200, non-trivial size
```

GitHub Pages rebuilds take 1–3 min. If a change is not live yet, **poll before assuming
failure** — a stale response is the usual cause, not a broken build.

---

## Conventions

- **Markdown pages** (`_pages/*.md`) use raw HTML for cards — the theme provides no markdown
  component syntax. Match the existing `<div class="…">` markup.
- Page front matter: `layout: single`, `author_profile: true`, `classes: wide`, explicit
  `permalink:`. **`permalink` is required** — `include: - _pages` in `_config.yml` means a page
  without one can be published at a wrong path.
- `layout: single` + `author_profile: true` renders the author sidebar
  (`_layouts/single.html` → `_includes/sidebar.html` → `_includes/author-profile.html`).
- The sidebar CV button lives in `_includes/author-profile-custom-links.html`.
- The footer (`_includes/footer.html`) has no link list — add navigation there only if
  deliberately wanted.
- `include: - .htaccess` and `- files` in `_config.yml` are no-ops (no `.htaccess`; `files/` is
  empty). Harmless.
- Posts live in `_posts/` with `YYYY-MM-DD-slug.md`. Front matter sets `toc: true`,
  `read_time: true`, `author_profile: true`; `related: true` and `share: true` come from
  `_config.yml` defaults.
