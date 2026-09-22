import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PACKAGE_DIR / "scripts"

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is optional for imported modules.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PACKAGE_DIR / ".env", override=False)


def env_path(name: str, default: str | Path) -> Path:
    """Return a path from an environment variable or a relative default."""

    value = os.environ.get(name)
    path = Path(value).expanduser() if value else Path(default)
    return path if path.is_absolute() else (PACKAGE_DIR / path)


DATA_DIR = env_path("OCCUBED_DATA_DIR", "data")
CACHE_DIR = env_path("OCCUBED_CACHE_DIR", ".cache")
ARTIFACTS_DIR = env_path("OCCUBED_ARTIFACTS_DIR", "artifacts")
FIGURE_DIR = env_path("OCCUBED_FIGURE_DIR", "figures")
SWEEP_ROOT = env_path("OCCUBED_SWEEP_ROOT", ARTIFACTS_DIR / "real_data_sweeps")
DEFAULT_SWEEP_DIR = env_path(
    "OCCUBED_SWEEP_DIR",
    SWEEP_ROOT / "multi_dataset_sweep_20260428_002833",
)
ACOUSTIC_DATA_DIR = env_path(
    "OCCUBED_ACOUSTIC_DATA_DIR",
    DATA_DIR / "acoustic_forest_soundscape",
)
IWILDCAM2022_DIR = env_path("OCCUBED_IWILDCAM2022_DIR", DATA_DIR / "iwildcam2022")
IWILDCAM_RAW_DIR = env_path(
    "OCCUBED_IWILDCAM_RAW_DIR",
    DATA_DIR / "iwildcam_unzipped",
)
