# ACORN: Active Continuous-Score Occupancy Modeling

## Codebase structure
Shared code and utilities are located in `src/occubed`, individual processing steps can be found in `scripts`, unit tests can be found in `tests`, and documentation can be found in `docs`.

## Setup
1. Copy `.env.example` to `.env`, fill in dataset, artifact, cache, and slurm configuration (set `SUBMITIT_DEBUG=1` to disable Slurm)
2. [Configure Google Earth Engine credentials](https://gee-documentation.readthedocs.io/en/latest/getting-started/index.html) to allow you to download environmental variables.
3. Scripts are declared as [uv](https://docs.astral.sh/uv/) script files that declare all of their dependencies and uses uv's `exclude-newer` configuration to improve reproducibility. To use this path, install `uv` and simply run scripts using `./scripts/<name>.py`.
4. Alternatively, you can run each script in your normal Python environment using `python ./scripts/<name>.py`, but you will then have to manage dependencies manually.

## Reproducing our experiments
1. Preprocess acoustic scores with `scripts/01_preprocess_acoustic.py`.
2. Preprocess iWildCam SpeciesNet scores with `scripts/02_preprocess_iwildcam.py prepare`, `run-shard`, or `submit`.
3. Run single-species experiments with `scripts/03_run_reviews.py` or sweep species with `scripts/03_submit_sweep.py`.
4. Aggregate results with `scripts/04_export_sweep.py`.
5. Produce analyses and figures with `scripts/05_*.py`.