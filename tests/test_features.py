"""Tests for feature engineering."""

from __future__ import annotations

from src.features.engineering import engineer_features


def test_engineer_features_creates_expected_columns(sample_hr_dataframe):
    cfg = {
        "career_stage_bins": {"Early": [0, 3], "Mid": [4, 10], "Senior": [11, 100]},
        "tenure_band_bins": {
            "New": [0, 2],
            "Established": [3, 5],
            "Veteran": [6, 10],
            "LongTenure": [11, 100],
        },
    }
    result = engineer_features(sample_hr_dataframe, cfg)

    expected = {
        "IncomePerExperience",
        "PromotionGap",
        "CareerStage",
        "CompensationGrowth",
        "SatisfactionIndex",
        "OvertimeRisk",
        "TenureBand",
        "CareerVelocity",
    }
    assert expected.issubset(set(result.columns))


def test_income_per_experience_formula(sample_hr_dataframe):
    cfg = {}
    result = engineer_features(sample_hr_dataframe, cfg)
    row = result.iloc[0]
    expected = row["MonthlyIncome"] / (row["TotalWorkingYears"] + 1)
    assert row["IncomePerExperience"] == expected


def test_career_stage_uses_fixed_bins(sample_hr_dataframe):
    cfg = {
        "career_stage_bins": {"Early": [0, 3], "Mid": [4, 10], "Senior": [11, 100]},
    }
    result = engineer_features(sample_hr_dataframe, cfg)
    assert result["CareerStage"].isin(["Early", "Mid", "Senior", "Unknown"]).all()
