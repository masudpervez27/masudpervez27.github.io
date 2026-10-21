source "https://rubygems.org"

# The plugin set is pinned to EXACTLY what GitHub Pages itself runs
# (github-pages 232 → jekyll 3.10.0, liquid 4.0.4, kramdown 2.4.0,
# jekyll-redirect-from 0.16.0, jekyll-include-cache 0.2.1, jekyll-feed 0.17.0,
# jekyll-sitemap 1.4.0, jekyll-gist 1.5.0, jekyll-paginate 1.1.0, jemoji 0.13.0).
# A local build is therefore a faithful preview of production.
#
# The `github-pages` meta-gem is intentionally NOT used. It transitively requires
# `jekyll-commonmark-ghpages` → `commonmarker ~> 0.22`, whose gemspec caps Ruby at
# `< 4.0`, so the meta-gem cannot resolve on a modern local Ruby. This site uses
# kramdown (`markdown: kramdown` in _config.yml), never commonmark, and the
# theme is vendored locally, so the individual plugins below are a complete
# substitute. The `whitelist:` list in _config.yml is what actually governs
# production; the meta-gem adds nothing there.
gem "jekyll", "3.10.0"
gem "liquid", "4.0.4"
gem "kramdown", "2.4.0"
gem "kramdown-parser-gfm", "1.1.0"
gem "rouge", "3.30.0"
gem "jekyll-remote-theme", "0.4.3"

# Plugins — the same list as `plugins:` in _config.yml
gem "jekyll-paginate", "1.1.0"
gem "jekyll-sitemap", "1.4.0"
gem "jekyll-gist", "1.5.0"
gem "jekyll-feed", "0.17.0"
gem "jekyll-include-cache", "0.2.1"
gem "jekyll-redirect-from", "0.16.0"
gem "jemoji", "0.13.0"

gem "webrick", "~> 1.7"

# Ruby 3.4 moved a batch of stdlib gems out of "default gems" into "bundled gems".
# Bundler hides bundled gems unless they are declared here, so Jekyll + Liquid
# fail to load on Ruby >= 3.4 without them.
#
# All of these are pure-Ruby and require only Ruby >= 2.5, so they are also safe for
# GitHub Pages' own build environment (which ignores this Gemfile entirely).
gem "csv"
gem "base64"
gem "bigdecimal"
gem "mutex_m"
gem "observer"
gem "abbrev"
gem "nkf"
gem "drb"
gem "getoptlong"
gem "ostruct"
