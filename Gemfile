source "https://rubygems.org"

gem "github-pages", group: :jekyll_plugins
gem "jekyll-include-cache", group: :jekyll_plugins
gem "jemoji", group: :jekyll_plugins
gem "webrick", "~> 1.7"

# Ruby 3.4 moved a batch of stdlib gems out of "default gems" into "bundled gems".
# Bundler hides bundled gems unless they are declared here, so Jekyll 3.9 + Liquid 4
# fail to load on Ruby >= 3.4 without them.
#
# All of these are pure-Ruby and require only Ruby >= 2.5, so they are also safe for
# GitHub Pages' own build environment (which ignores this Gemfile entirely).
gem "csv"
gem "bigdecimal"
gem "mutex_m"
gem "observer"
gem "abbrev"
gem "nkf"
gem "drb"
gem "getoptlong"
gem "ostruct"
