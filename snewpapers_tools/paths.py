"""Where this repo keeps downloaded and cached data (gitignored)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
