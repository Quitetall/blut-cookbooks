# BLUT standard and core cookbooks

Reusable training building blocks for the [BLUT](https://github.com/Quitetall/blut)
workflow engine.

This repository owns two Rust crates and one Python distribution:

- `blut-cookbook-standard`: standard data, SFT, conversion, registration, and
  concrete trainer-backend stages.
- `blut-cookbook-core`: generic ingredient-driven training and evaluation
  recipes. Its binary is `blut-core`.
- `blut-cookbook-standard` on PyPI: lightweight `blut_core` catalog/provenance
  package plus runnable `blut_standard` trainer modules.

## Status

`0.2.0-alpha.1` is a preview. Public registries contain only stages and recipes
that perform real work. DPO/distillation shims, synthetic evaluators, the
metadata-only LoRA merge stage, and the recipe that depends on that merge remain
importable for source compatibility but are not registered by default.

## Install

Rust applications should pin the preview exactly:

```toml
[dependencies]
blut-cookbook-standard = "=0.2.0-alpha.1"
blut-cookbook-core = "=0.2.0-alpha.1"
```

Install the Python runtime and its declared training dependencies into the
interpreter used by BLUT:

```sh
python3.12 -m pip install "blut-cookbook-standard[training]==0.2.0a1"
python3.12 -c "import blut_core; print(blut_core.__file__)"
```

`training` installs PyTorch, Transformers, and dataset support. Omit the extra
for lightweight catalog/provenance tooling or when those packages are managed
by a site-specific CUDA environment. Python 3.12 is the supported runtime for
this preview. Rust MSRV is 1.88.

## Development

During coordinated pre-publication development, patch `blut` to a reviewed
local checkout without changing this repository's manifests:

```sh
cargo test --workspace --all-targets \
  --config 'patch.crates-io.blut.path="../blut"'
python3.12 -m pip install -e '.[test]'
python3.12 -m pytest -q python
```

Before release, run format, clippy, tests, rustdoc, wheel inspection, and crate
package checks from a clean checkout. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security and telemetry

Training consumes code, models, and datasets from outside this repository.
Treat all artifacts as untrusted until verified. Report vulnerabilities using
[SECURITY.md](SECURITY.md). BLUT does not enable external telemetry by default;
downstream cookbooks must require explicit user opt-in.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
