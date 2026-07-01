"""Project paths and constants for Phase 1 notebooks."""

from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root() -> Path:
    """Resolve repository root regardless of notebook or CLI working directory."""
    anchors = [Path.cwd()]
    if Path.cwd().name == "notebooks":
        anchors.append(Path.cwd().parent)

    file_anchor = Path(__file__).resolve().parent
    for _ in range(5):
        anchors.append(file_anchor)
        file_anchor = file_anchor.parent

    for candidate in anchors:
        if (candidate / "data" / "raw").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODEL_DIR = PROJECT_ROOT / "models" / "phase1"

RANDOM_STATE = 42
TARGET = "Attrition"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"
TEST_SIZE = 0.20

DROP_COLS = ["EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"]

ATTRITION_COLORS = {"No": "#2E86AB", "Yes": "#E94F37"}

LIKERT_COLS = [
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "JobInvolvement",
    "PerformanceRating",
]
SATISFACTION_INDEX_COLS = LIKERT_COLS[:4]

ORIGINAL_CATEGORICAL = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]
ENGINEERED_CATEGORICAL = ["CareerStage", "TenureBand"]

FEATURE_CFG = {
    "career_stage_bins": {"Early": [0, 3], "Mid": [4, 10], "Senior": [11, 100]},
    "tenure_band_bins": {
        "New": [0, 2],
        "Established": [3, 5],
        "Veteran": [6, 10],
        "LongTenure": [11, 100],
    },
}
