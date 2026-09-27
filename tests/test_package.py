import json
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from importlib import resources
from pathlib import Path

import dist_dual_input_stream_transformer as dist_pkg


ROOT = Path(__file__).resolve().parents[1]


def test_version_is_exposed():
    assert dist_pkg.__version__ == "0.1.0"


def test_bundled_algorithm_config_is_valid_json():
    config = resources.files("dist_dual_input_stream_transformer").joinpath("data/algo_cfgs_all.json")
    data = json.loads(config.read_text(encoding="utf-8"))
    assert "warp" in data
    assert "slice" in data


def test_streamlit_is_optional():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base_dependencies = pyproject["project"]["dependencies"]
    app_dependencies = pyproject["project"]["optional-dependencies"]["app"]

    assert not any(dep.lower().startswith("streamlit") for dep in base_dependencies)
    assert not any(dep.lower().startswith("stqdm") for dep in base_dependencies)
    assert any(dep.lower().startswith("streamlit") for dep in app_dependencies)
    assert any(dep.lower().startswith("stqdm") for dep in app_dependencies)


def test_console_entry_points_are_declared():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["dist-app"] == "dist_dual_input_stream_transformer.cli:app"
    assert scripts["dist-train"] == "dist_dual_input_stream_transformer.train:cli"
