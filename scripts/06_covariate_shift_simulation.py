#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "occubed",
#   "biolith==0.1.0",
#   "fire",
#   "jax==0.6.2",
#   "jaxlib==0.6.2",
#   "matplotlib",
#   "numpy",
#   "numpyro==0.19.0",
#   "pandas",
# ]
#
# [tool.uv]
# exclude-newer = "2026-05-01T00:00:00Z"
#
# [tool.uv.sources]
# occubed = { path = ".." }
# ///
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from occubed import covariate_shift_simulation as target  # noqa: I001


if __name__ == "__main__":
    fire.Fire(target.main)
