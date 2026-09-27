"""Repository-root launcher for the optional DIST Streamlit application.

This keeps the familiar ``streamlit run app.py`` workflow for cloned repos,
while the actual application lives inside the installable package.
"""

from pathlib import Path
import runpy
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import streamlit  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        'The Streamlit app is optional. Install it with `uv pip install -e ".[app]"` '
        '(or `pip install -e ".[app]"`), then run `streamlit run app.py`.'
    ) from exc

runpy.run_module("dist_dual_input_stream_transformer.app", run_name="__main__")
