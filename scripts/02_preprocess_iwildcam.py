#!/usr/bin/env -S uv run --no-project --python 3.11 --script
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "occubed",
#   "fire",
#   "kagglehub",
#   "numpy<2.0",
#   "pandas",
#   "pillow",
#   "python-dotenv",
#   "submitit",
#   "tensorflow[and-cuda]==2.15.1",
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

from occubed import preprocess_iwildcam

if __name__ == "__main__":
    fire.Fire(
        {
            "prepare": preprocess_iwildcam.prepare,
            "run-shard": preprocess_iwildcam.run_shard,
            "run_shard": preprocess_iwildcam.run_shard,
            "submit": preprocess_iwildcam.submit,
        }
    )
