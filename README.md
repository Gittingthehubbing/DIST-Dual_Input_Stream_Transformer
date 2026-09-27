# DIST: Dual Input Stream Transformer

Official repo for [Dual input stream transformer for eye-tracking line assignment](https://arxiv.org/abs/2311.06095).

## Install as a Python package

DIST now uses a modern `pyproject.toml` and a `src/` package layout. Python 3.11 is the recommended environment for the research code; package metadata currently allows Python 3.10+.

From a clone, using [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.11
```

Activate the environment and install the library:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
uv pip install -e .
```

**macOS/Linux**

```sh
source .venv/bin/activate
uv pip install -e .
```

A normal package install deliberately **does not install Streamlit**. This keeps the dependency set smaller for projects that only want to import DIST, such as downstream integrations.

The repository can also be installed directly from GitHub:

```sh
pip install "git+https://github.com/Gittingthehubbing/DIST-Dual_Input_Stream_Transformer.git"
```

The package is imported through the namespace:

```python
from dist_dual_input_stream_transformer import models
from dist_dual_input_stream_transformer import utils as ut

classic_cfg = ut.get_classic_cfg()
```

The old root-level modules are retained as lightweight compatibility wrappers for existing cloned-repository scripts and notebooks. New code should prefer the namespaced imports above.

## Optional Streamlit app

Install the app dependencies explicitly:

```sh
uv pip install -e ".[app]"
```

Then either use the installed console command:

```sh
dist-app
```

or keep the familiar repository workflow:

```sh
streamlit run app.py
```

The app expects model checkpoints/configuration files in a `models/` directory relative to the directory from which it is launched, matching the original repository behaviour. Model files are not bundled into the Python wheel.

## Download data

To get the data that was used to develop and train the models please see: [OSF Link](https://osf.io/zt9gn).

## Run via Hugging Face Space

For an easy way of applying the model to `.asc` files or the preprocessed files linked above, see the [Hugging Face Space](https://huggingface.co/spaces/bugroup/Eye_Tracking_Drift_Correction).

## Run in a notebook

Install the notebook tooling alongside DIST:

```sh
uv pip install -e ".[notebook]"
uv run jupyter lab
```

The root `run_in_notebook.ipynb` and the copy under `examples/` use the installed package namespace.

## Run training

Training has one additional optional dependency (`tensorboard`):

```sh
uv pip install -e ".[train]"
```

To train a DIST model, adjust the contents of `experiments/main.yaml` to your system and data location. For example, if you have downloaded and unzipped the data from the [OSF Link](https://osf.io/zt9gn) (keeping the folder structure) to `F:/pydata/ET2/`, the original example configuration can be adapted from there.

`dataset_folder_idx_training` determines which datasets are used to train on and `use_reduced_set` determines whether the run uses only a few samples or the full set.

You can run training with the installed command:

```sh
dist-train experiments/main.yaml
```

or use the backwards-compatible repository launcher:

```sh
python train.py
```

The results are saved under `results/`.

## Full legacy-style environment

For users following the repository's former `requirements.txt` workflow, this still works and installs the library plus the optional app and training dependencies:

```sh
pip install -r requirements.txt
```

For everything including notebook tooling:

```sh
uv pip install -e ".[full]"
```

## Development and packaging

Install development tooling with:

```sh
uv pip install -e ".[dev]"
```

Run the package tests:

```sh
uv run pytest
```

Build a wheel and source distribution:

```sh
uv build
```

Validate distributions before publishing with:

```sh
uv run twine check dist/*
```

## Licensing

The repository currently has no `LICENSE` file. The packaging changes do not choose a license on behalf of the authors. Add the intended license separately if you want to grant explicit redistribution/modification rights or publish with corresponding license metadata.
