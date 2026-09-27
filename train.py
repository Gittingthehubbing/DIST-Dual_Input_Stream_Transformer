"""Backward-compatible launcher for DIST training.

The implementation lives in :mod:`dist_dual_input_stream_transformer.train`.
"""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dist_dual_input_stream_transformer.train import *  # noqa: F401,F403,E402
from dist_dual_input_stream_transformer.train import cli as _cli  # noqa: E402


if __name__ == "__main__":
    _cli()
