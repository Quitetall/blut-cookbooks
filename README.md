# BLUT Backends

Owner repository for reusable BLUT cookbook implementations:

- `blut-cookbook-standard`: generic LLM stages and concrete trainer backends.
- `blut-cookbook-core`: generic ML stages, recipes, and Python ingredients.

Production manifests pin the BLUT engine by Git revision. LamQuant may provide
untracked local Cargo overrides for editable development, but this repository
builds without sibling checkouts.

```sh
cargo test --workspace --all-targets --locked
```
