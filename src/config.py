from pathlib import Path

# ==========================================================
# Project Root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# Data Directories
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA = DATA_DIR / "raw"
INTERIM_DATA = DATA_DIR / "interim"
PROCESSED_DATA = DATA_DIR / "processed"
FEATURE_DATA = DATA_DIR / "features"

# ==========================================================
# Outputs
# ==========================================================

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = PROJECT_ROOT / "figures"
LOG_DIR = PROJECT_ROOT / "logs"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models"

# ==========================================================
# External Dataset Location
# ==========================================================

UOM_ROOT = Path(
    r"T:\UOM\sharpic-ManchesterCSCoordinatedDiabetesStudy-a9e8025"
)

# ==========================================================
# Create directories automatically
# ==========================================================

for directory in [
    DATA_DIR,
    RAW_DATA,
    INTERIM_DATA,
    PROCESSED_DATA,
    FEATURE_DATA,
    REPORT_DIR,
    FIGURE_DIR,
    LOG_DIR,
    ARTIFACT_DIR,
    MODEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)