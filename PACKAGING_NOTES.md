# Packaging notes

This change turns the flat research repository into a conventional installable `src/`-layout Python distribution while preserving the familiar cloned-repository entry points.

## What changed

- Distribution name: `dist-dual-input-stream-transformer`.
- Import package: `dist_dual_input_stream_transformer`.
- Added `pyproject.toml` using PEP 517/621 metadata and `setuptools.build_meta`.
- Declared Python `>=3.10`; Python 3.11 remains the recommended baseline until CI establishes a wider compatibility matrix.
- Added `dist-app` and `dist-train` console entry points.
- Moved the canonical Python implementation under `src/dist_dual_input_stream_transformer/` and rewrote internal imports to use that namespace.
- Kept lightweight root-level compatibility wrappers so existing `import utils`, `import models`, `python train.py`, and `streamlit run app.py` workflows continue to work from a clone without maintaining two copies of the implementation.
- Bundled `algo_cfgs_all.json` as package data. `utils.get_classic_cfg()` now uses the packaged resource by default, so installed code no longer depends on the current working directory for that file.
- Removed import-time CLI argument parsing from the canonical `train.py`; arguments are parsed only when training is invoked.
- Updated both notebook copies to import the namespaced package.
- Added lightweight package/build tests.

## Optional dependency split

A normal install such as:

```sh
pip install .
```

or:

```sh
pip install "git+https://github.com/Gittingthehubbing/DIST-Dual_Input_Stream_Transformer.git"
```

does **not** install Streamlit.

Optional extras are:

- `app`: `streamlit` and `stqdm`.
- `train`: `tensorboard`.
- `notebook`: Jupyter/IPython kernel tooling.
- `full`: app + training + notebook extras.
- `dev`: build, test, lint, and publishing tools.

`utils.py` historically mixes general helpers with some Streamlit session-state integration. It now treats Streamlit as optional at import time so model/library users can import the package without the `app` extra. Functions that specifically need Streamlit still require the `app` extra when used in that mode.

The existing `requirements.txt` is retained as a compatibility path and installs `.[app,train]`, reproducing the former all-in-one repository environment while leaving a plain package install leaner.

## Dependencies

The original dependency set is preserved apart from moving `streamlit`, `stqdm`, and `tensorboard` to appropriate optional extras. Direct dependencies imported by the source but missing from the old requirements file are declared explicitly: `Pillow`, `requests`, `scipy`, and `torchvision`.

## Not included / not decided here

- Pretrained model checkpoints and model configuration artifacts remain in the repository's separate `models/` directory and are not embedded in the wheel.
- Experiment YAMLs remain in the repository's `experiments/` directory and are not embedded in the wheel.
- No license was invented; the repository currently has no `LICENSE` file.
- No `uv.lock` is committed. For a library, `pyproject.toml` is the interoperability contract; a maintainer lockfile can be added later if desired.

Before a public PyPI release, confirm the final distribution name on PyPI, add the intended license, and decide whether version `0.1.0` matches the project's release/versioning policy.
