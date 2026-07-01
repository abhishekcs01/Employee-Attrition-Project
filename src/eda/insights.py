"""Business insights generation from EDA results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _attrition_rate_by_column(
    df: pd.DataFrame, col: str, target: str, positive: str = "Yes"
) -> pd.Series:
    """Calculate attrition rate per category."""
    rates = df.groupby(col)[target].apply(lambda x: (x == positive).mean() * 100)
    return rates.sort_values(ascending=False)


def generate_business_insights(
    df: pd.DataFrame,
    target_column: str,
    cfg: dict[str, Any],
    imbalance_stats: dict[str, Any],
    outlier_summary: pd.DataFrame | None = None,
) -> str:
    """Generate markdown business insights from statistical analysis."""
    insights_cfg = cfg.get("insights", {})
    top_n = int(insights_cfg.get("top_n_categories", 5))
    satisfaction_cols = insights_cfg.get(
        "satisfaction_columns",
        [
            "EnvironmentSatisfaction",
            "JobSatisfaction",
            "RelationshipSatisfaction",
            "WorkLifeBalance",
        ],
    )
    positive_class = cfg.get("target_analysis", {}).get("positive_class", "Yes")

    lines: list[str] = [
        "# Business Insights - IBM HR Employee Attrition",
        "",
        "## Executive Summary",
        "",
        f"- **Total employees analyzed:** {len(df):,}",
        f"- **Overall attrition rate:** {imbalance_stats.get('positive_rate_pct', 0):.1f}%",
        f"- **Class imbalance ratio:** {imbalance_stats.get('imbalance_ratio', 'N/A')}:1",
        "",
    ]

    # Department insights
    if "Department" in df.columns:
        dept_rates = _attrition_rate_by_column(df, "Department", target_column, positive_class)
        lines.extend(
            [
                "## Attrition by Department",
                "",
                "| Department | Attrition Rate (%) |",
                "|------------|-------------------|",
            ]
        )
        for dept, rate in dept_rates.items():
            lines.append(f"| {dept} | {rate:.1f} |")
        highest_dept = dept_rates.index[0]
        lines.extend(
            [
                "",
                f"**Insight:** `{highest_dept}` has the highest attrition rate "
                f"({dept_rates.iloc[0]:.1f}%). HR should prioritize retention "
                f"programs in this department.",
                "",
            ]
        )

    # Overtime insights
    if "OverTime" in df.columns:
        ot_rates = _attrition_rate_by_column(df, "OverTime", target_column, positive_class)
        lines.extend(
            [
                "## Overtime Impact",
                "",
                "| OverTime | Attrition Rate (%) |",
                "|----------|-------------------|",
            ]
        )
        for ot, rate in ot_rates.items():
            lines.append(f"| {ot} | {rate:.1f} |")
        if "Yes" in ot_rates.index and "No" in ot_rates.index:
            diff = ot_rates.get("Yes", 0) - ot_rates.get("No", 0)
            lines.extend(
                [
                    "",
                    f"**Insight:** Employees working overtime have "
                    f"{'higher' if diff > 0 else 'lower'} attrition "
                    f"({abs(diff):.1f} percentage points difference). "
                    "Workload management and overtime policies should be reviewed.",
                    "",
                ]
            )

    # Job role insights
    if "JobRole" in df.columns:
        role_rates = _attrition_rate_by_column(df, "JobRole", target_column, positive_class)
        top_roles = role_rates.head(top_n)
        lines.extend(
            [
                f"## Top {top_n} Job Roles by Attrition Rate",
                "",
                "| Job Role | Attrition Rate (%) |",
                "|----------|-------------------|",
            ]
        )
        for role, rate in top_roles.items():
            lines.append(f"| {role} | {rate:.1f} |")
        lines.extend(
            [
                "",
                f"**Insight:** `{top_roles.index[0]}` role shows the highest attrition risk. "
                "Targeted career development and compensation review recommended.",
                "",
            ]
        )

    # Income analysis
    if "MonthlyIncome" in df.columns:
        income_by_attrition = df.groupby(target_column)["MonthlyIncome"].agg(
            ["mean", "median", "std"]
        )
        lines.extend(
            [
                "## Compensation Analysis",
                "",
                "| Attrition | Mean Income | Median Income | Std Dev |",
                "|-----------|-------------|---------------|---------|",
            ]
        )
        for idx, row in income_by_attrition.iterrows():
            lines.append(
                f"| {idx} | ${row['mean']:,.0f} | ${row['median']:,.0f} | ${row['std']:,.0f} |"
            )
        lines.extend(
            [
                "",
                "**Insight:** Lower-income employees may have higher attrition propensity. "
                "Competitive compensation benchmarking is advised.",
                "",
            ]
        )

    # Tenure patterns
    tenure_cols = ["YearsAtCompany", "YearsSinceLastPromotion", "YearsInCurrentRole"]
    available_tenure = [c for c in tenure_cols if c in df.columns]
    if available_tenure:
        lines.extend(["## Tenure Patterns", ""])
        for col in available_tenure:
            mean_by_attrition = df.groupby(target_column)[col].mean()
            stayers_avg = mean_by_attrition.drop(positive_class, errors="ignore").mean()
            lines.append(
                f"- **{col}:** Leavers avg {mean_by_attrition.get(positive_class, 0):.1f} yrs "
                f"vs Stayers avg {stayers_avg:.1f} yrs"
            )
        lines.extend(
            [
                "",
                "**Insight:** Employees who leave tend to have shorter tenure and longer time "
                "since last promotion. Regular promotion cycles and early-career engagement "
                "are critical.",
                "",
            ]
        )

    # Satisfaction composite
    available_sat = [c for c in satisfaction_cols if c in df.columns]
    if available_sat:
        df_sat = df.copy()
        df_sat["SatisfactionComposite"] = df_sat[available_sat].mean(axis=1)
        sat_by_attrition = df_sat.groupby(target_column)["SatisfactionComposite"].mean()
        lines.extend(
            [
                "## Satisfaction Index",
                "",
                "| Attrition | Avg Satisfaction (1-4 scale) |",
                "|-----------|---------------------------|",
            ]
        )
        for idx, val in sat_by_attrition.items():
            lines.append(f"| {idx} | {val:.2f} |")
        lines.extend(
            [
                "",
                "**Insight:** Lower satisfaction scores correlate with attrition. "
                "Employee engagement surveys and manager effectiveness programs are recommended.",
                "",
            ]
        )

    # Outlier summary
    if outlier_summary is not None and not outlier_summary.empty:
        high_outlier = outlier_summary.nlargest(3, "outlier_pct")
        lines.extend(
            [
                "## Data Quality - Outlier Features",
                "",
                "Features with highest outlier percentage:",
                "",
            ]
        )
        for _, row in high_outlier.iterrows():
            lines.append(f"- `{row['feature']}`: {row['outlier_pct']}% outliers")
        lines.append("")

    lines.extend(
        [
            "## Recommended Actions",
            "",
            "1. Implement retention programs for high-risk departments and job roles.",
            "2. Review overtime policies and workload distribution.",
            "3. Conduct compensation benchmarking for below-market employees.",
            "4. Establish structured promotion timelines (address promotion gaps).",
            "5. Deploy regular satisfaction pulse surveys with manager accountability.",
            "6. Use predictive modeling (next phases) to identify at-risk employees proactively.",
            "",
        ]
    )

    return "\n".join(lines)


def save_business_insights(content: str, output_path: Path) -> Path:
    """Write business insights markdown to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    logger.info("Business insights saved to %s", output_path)
    return output_path
