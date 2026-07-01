"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture
def project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def schema_config(project_root: Path) -> dict:
    """Load schema configuration."""
    schema_path = project_root / "configs" / "data" / "schema.yaml"
    with schema_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_hr_dataframe(schema_config: dict) -> pd.DataFrame:
    """Create minimal valid IBM HR dataset for testing."""
    records = [
        {
            "Age": 41,
            "Attrition": "Yes",
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 1102,
            "Department": "Sales",
            "DistanceFromHome": 1,
            "Education": 2,
            "EducationField": "Life Sciences",
            "EmployeeCount": 1,
            "EmployeeNumber": 1,
            "EnvironmentSatisfaction": 2,
            "Gender": "Female",
            "HourlyRate": 94,
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobRole": "Sales Executive",
            "JobSatisfaction": 4,
            "MaritalStatus": "Single",
            "MonthlyIncome": 5993,
            "MonthlyRate": 19479,
            "NumCompaniesWorked": 8,
            "Over18": "Y",
            "OverTime": "Yes",
            "PercentSalaryHike": 11,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 1,
            "StandardHours": 80,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 8,
            "TrainingTimesLastYear": 0,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 6,
            "YearsInCurrentRole": 4,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 5,
        },
        {
            "Age": 49,
            "Attrition": "No",
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 279,
            "Department": "Research & Development",
            "DistanceFromHome": 8,
            "Education": 1,
            "EducationField": "Life Sciences",
            "EmployeeCount": 1,
            "EmployeeNumber": 2,
            "EnvironmentSatisfaction": 3,
            "Gender": "Male",
            "HourlyRate": 61,
            "JobInvolvement": 2,
            "JobLevel": 2,
            "JobRole": "Research Scientist",
            "JobSatisfaction": 2,
            "MaritalStatus": "Married",
            "MonthlyIncome": 5130,
            "MonthlyRate": 24907,
            "NumCompaniesWorked": 1,
            "Over18": "Y",
            "OverTime": "No",
            "PercentSalaryHike": 23,
            "PerformanceRating": 4,
            "RelationshipSatisfaction": 4,
            "StandardHours": 80,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 10,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 10,
            "YearsInCurrentRole": 7,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 7,
        },
        {
            "Age": 37,
            "Attrition": "Yes",
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 1373,
            "Department": "Research & Development",
            "DistanceFromHome": 2,
            "Education": 2,
            "EducationField": "Other",
            "EmployeeCount": 1,
            "EmployeeNumber": 3,
            "EnvironmentSatisfaction": 4,
            "Gender": "Male",
            "HourlyRate": 92,
            "JobInvolvement": 3,
            "JobLevel": 1,
            "JobRole": "Laboratory Technician",
            "JobSatisfaction": 3,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2090,
            "MonthlyRate": 2396,
            "NumCompaniesWorked": 6,
            "Over18": "Y",
            "OverTime": "Yes",
            "PercentSalaryHike": 15,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StandardHours": 80,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 7,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 0,
            "YearsInCurrentRole": 0,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0,
        },
    ]
    return pd.DataFrame(records)


@pytest.fixture
def expanded_hr_dataframe(sample_hr_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Expanded dataset for preprocessing split tests."""
    return pd.concat([sample_hr_dataframe] * 10, ignore_index=True)
