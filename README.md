# BLUT Backends

Owner repository for reusable BLUT cookbook implementations:

- `blut-cookbook-standard`: generic LLM stages and concrete trainer backends.
- `blut-cookbook-core`: generic ML stages, recipes, and Python ingredients.

Production manifests pin the BLUT engine by Git revision. LamQuant may provide
untracked local Cargo overrides for editable development, but this repository
builds without sibling checkouts.

```sh
cargo test --workspace --all-targets --locked
PYTHONPATH=python python -m pytest -q python
```

Build the production cookbook image with a BuildKit secret when the pinned
BLUT engine revision is private. The token is available only to the build
step that resolves the engine and builds its matching Starlark evaluator:

```sh
GITHUB_TOKEN="$(gh auth token)" docker build \
  --secret id=github_token,env=GITHUB_TOKEN \
  --file docker/Dockerfile.blut-core \
  --tag blut-core:local .
```

The image pins its base layers, engine revision, `tini` package, and Rust lock.
The optional HF trainer provisions from an embedded hash-locked Python
3.12/Linux x86-64 resolution; refresh it explicitly with the command recorded
in `src/backends/hf_trainer/requirements-hf.lock`, then validate that lock in a
clean Python 3.12 environment before committing it. Provisioning prefers
`python3.12`; set `BLUT_HF_PYTHON` when that interpreter has a custom path.

`python/blut_core` is the single Python runtime package for both crates. Runtime
launchers must install it or prepend this repository's `python` directory to
`PYTHONPATH`.
