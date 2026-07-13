# Changelog

All notable public changes are recorded here. This project follows Semantic
Versioning; preview versions may change APIs between prereleases.

## 0.2.0-alpha.1 - Unreleased

- Establish `blut-cookbook-standard` and `blut-cookbook-core` as one versioned
  release train on BLUT `0.2.0-alpha.1`.
- Add installable Python 3.12 distribution for `blut_core`.
- Package runnable standard trainer modules and embed the HF trainer wrapper so
  installed crates do not depend on a source checkout.
- Propagate `blut-core` CLI failures through process exit status.
- Exclude stub, synthetic, and metadata-only stages from default registries.
- Persist WSD scheduler progress with fail-closed resume compatibility checks.
- Store HF job specifications with owner-only permissions on Unix.
- Add release CI, security policy, package allowlists, and AGPL licensing.
