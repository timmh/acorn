#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "occubed",
#   "fire",
#   "numpy",
#   "pandas",
#   "perch-hoplite",
#   "soundfile",
# ]
#
# [tool.uv]
# exclude-newer = "2026-05-01T00:00:00Z"
#
# [tool.uv.sources]
# occubed = { path = ".." }
# ///
import fire

from occubed import preprocess_acoustic

if __name__ == "__main__":
    fire.Fire(preprocess_acoustic.main)
