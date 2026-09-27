"""Console entry points for DIST."""

from __future__ import annotations

import sys
from pathlib import Path


_APP_INSTALL_HINT = (
    "The DIST Streamlit app is an optional dependency. Install it with "
    "`pip install \"dist-dual-input-stream-transformer[app]\"` (or `uv pip install -e \".[app]\"` "
    "from a clone), then run `dist-app` again."
)


def app() -> None:
    """Launch the bundled Streamlit application."""
    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError as exc:
        raise SystemExit(_APP_INSTALL_HINT) from exc

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(stcli.main())
