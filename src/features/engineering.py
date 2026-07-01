"""Feature engineering for employee attrition prediction."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_DEFINITIONS = {
    "IncomePerExperience": "MonthlyIncome divided by (TotalWorkingYears + 1).",
    "PromotionGap": "YearsAtCompany minus YearsSinceLastPromotion.",
    "CareerStage": "Early (0-3 yrs), Mid (4-10 yrs), Senior (10+ yrs) from TotalWorkingYears.",
    "CompensationGrowth": "PercentSalaryHike multiplied by YearsAtCompany.",
    "SatisfactionIndex": (
        "Mean of EnvironmentSatisfaction, JobSatisfaction, "
        "RelationshipSatisfaction, WorkLifeBalance."
    ),
    "OvertimeRisk": "Binary: 1 if OverTime=Yes AND WorkLifeBalance <= 2, else 0.",
    "TenureBand": (
        "New (0-2), Established (3-5), Veteran (6-10), LongTenure (10+) from YearsAtCompany."
    ),
    "CareerVelocity": "JobLevel divided by (TotalWorkingYears + 1).",
}


def _assign_binned_category(
    series: pd.Series,
    bins: dict[str, list[int]],
) -> pd.Series:
    """Assign fixed domain bins (no data-driven quantiles)."""
    result = pd.Series(index=series.index, dtype="object")
    for label, (low, high) in bins.items():
        mask = (series >= low) & (series <= high)
        result[mask] = label
    return result.fillna("Unknown")


def engineer_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create engineered features using fixed domain rules."""
    engineered = df.copy()

    engineered["IncomePerExperience"] = engineered["MonthlyIncome"] / (
        engineered["TotalWorkingYears"] + 1
    )
    engineered["PromotionGap"] = (
        engineered["YearsAtCompany"] - engineered["YearsSinceLastPromotion"]
    )

    career_bins = cfg.get(
        "career_stage_bins", {"Early": [0, 3], "Mid": [4, 10], "Senior": [11, 100]}
    )
    engineered["CareerStage"] = _assign_binned_category(
        engineered["TotalWorkingYears"], career_bins
    )

    engineered["CompensationGrowth"] = (
        engineered["PercentSalaryHike"] * engineered["YearsAtCompany"]
    )

    satisfaction_cols = [
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "RelationshipSatisfaction",
        "WorkLifeBalance",
    ]
    engineered["SatisfactionIndex"] = engineered[satisfaction_cols].mean(axis=1)

    engineered["OvertimeRisk"] = (
        (engineered["OverTime"] == "Yes") & (engineered["WorkLifeBalance"] <= 2)
    ).astype(int)

    tenure_bins = cfg.get(
        "tenure_band_bins",
        {"New": [0, 2], "Established": [3, 5], "Veteran": [6, 10], "LongTenure": [11, 100]},
    )
    engineered["TenureBand"] = _assign_binned_category(engineered["YearsAtCompany"], tenure_bins)

    engineered["CareerVelocity"] = engineered["JobLevel"] / (engineered["TotalWorkingYears"] + 1)

    logger.info("Engineered %d features on %d rows", len(FEATURE_DEFINITIONS), len(engineered))
    return engineered


def write_feature_dictionary(path: Path) -> None:
    """Write feature documentation markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Feature Engineering Dictionary", ""]
    for name, description in FEATURE_DEFINITIONS.items():
        lines.extend([f"## {name}", "", description, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
