"""Feature definitions for Phase 1 notebooks (implementation lives in Notebook 03)."""

from __future__ import annotations

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
