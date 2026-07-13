# blut-cookbook-core

Generic ingredient-driven training and evaluation recipes for BLUT.

```toml
[dependencies]
blut-cookbook-core = "=0.2.0-alpha.1"
```

Runtime stages invoke Python 3.12 modules from the separately installable
`blut-cookbook-standard` Python distribution:

```sh
python3.12 -m pip install "blut-cookbook-standard[training]==0.2.0a1"
```

This preview registers `train_from_dataset` and `eval_only`. Experimental
recipes remain available in source but are not included in the default
registry until all stages produce real artifacts.

AGPL-3.0-or-later.
