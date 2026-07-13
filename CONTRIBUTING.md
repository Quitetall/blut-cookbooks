# Contributing

Use Rust 1.88+ and Python 3.12. Make changes from a clean checkout.

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
python3.12 -m pip install -e '.[test,build]'
python3.12 -m pytest -q python
python3.12 -m build
python3.12 -m twine check dist/*
```

Before BLUT `0.2.0-alpha.1` exists on crates.io, pass a temporary local Cargo
patch as shown in the README. Never commit machine-specific path patches.

New public stages need real outputs, fail-closed error handling, deterministic
fixtures where possible, and end-to-end tests. Synthetic metrics belong only in
tests and must never be presented as evaluation results.
