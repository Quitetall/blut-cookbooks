# blut-core

The shared Python runtime for [BLUT](https://github.com/Quitetall/blut) cookbooks:
domain-agnostic building blocks any cookbook can reuse.

- `runctx` — run identity and filesystem anchors from the BLUT environment
- `MetricLog` — atomic per-epoch CSV/Parquet metric writer
- `read_metric` — verbatim metric/log reader (no model in the path)
- `status` — the StatusUpdate wire protocol
- `RunManifest` — run provenance, written even on crash
- `checkpoint` — corruption-safe save/load and the resume payload contract
- `sysgauge` — best-effort GPU/host snapshot for the metric stream
- `async_io` — bounded, backpressured delivery for persistence work
- the ingredient specs and builders the generic training recipes build on

`torch` is imported lazily, inside `checkpoint` and `sysgauge` only, so
`import blut_core` stays cheap. Install `blut-core[torch]` if you call those.

Apache-2.0. Published from the `blut-cookbooks` repository.
