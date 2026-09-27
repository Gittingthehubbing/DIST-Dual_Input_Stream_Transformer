"""Backward-compatible wrapper for :mod:`dist_dual_input_stream_transformer.analysis_funcs`.

New code should import from the ``dist_dual_input_stream_transformer`` package.
This file remains so existing notebooks/scripts in a cloned repository continue
to work while the installable package uses the ``src/`` layout.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dist_dual_input_stream_transformer.analysis_funcs import *  # noqa: F401,F403,E402
