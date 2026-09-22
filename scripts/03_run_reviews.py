#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "occubed",
#   "biolith==0.1.0",
#   "contextily",
#   "earthengine-api",
#   "fire",
#   "geopandas",
#   "imageio-ffmpeg",
#   "ipython",
#   "jax",
#   "joblib",
#   "matplotlib",
#   "numpy",
#   "numpyro",
#   "pandas",
#   "pyogrio",
#   "pyproj",
#   "python-dotenv",
#   "submitit",
#   "tqdm",
# ]
#
# [tool.uv]
# exclude-newer = "2026-05-01T00:00:00Z"
#
# [tool.uv.sources]
# occubed = { path = ".." }
# ///
import fire

from occubed import run_reviews as target

if __name__ == "__main__":
    fire.Fire(target.main)
