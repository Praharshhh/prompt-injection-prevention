"""Initialize package."""
from pathlib import Path

# Package metadata
__version__ = "0.1.0"
__author__ = "Your Name"

# Set up package-level paths
PACKAGE_ROOT = Path(__file__).parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
