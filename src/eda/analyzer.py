"""EDA orchestrator - coordinates all exploratory analysis modules."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.eda.categorical import plot_categorical_frequencies
from src.eda.correlations import plot_correlation_heatmaps
from src.eda.distributions import plot_numerical_distributions
from src.eda.insights import generate_business_insights, save_business_insights
from src.eda.outliers import plot_outlier_boxplots
from src.eda.pairplots import plot_pairplot, select_top_correlated_features
from src.eda.profiler import generate_profile_report
from src.eda.schema_validator import ValidationResult
from src.eda.target_analysis import analyze_class_imbalance

logger = logging.getLogger(__name__)


@dataclass
class EDAResult:
    """Container for all EDA outputs and metadata."""

    output_dir: Path
    plots_dir: Path
    validation_result: ValidationResult
    imbalance_stats: dict[str, Any]
    plot_paths: list[Path] = field(default_factory=list)
    profile_report_path: Path | None = None
    business_insights_path: Path | None = None
    summary_path: Path | None = None

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize EDA results to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "plots_dir": str(self.plots_dir),
            "validation": self.validation_result.to_dict(),
            "imbalance_stats": self.imbalance_stats,
            "plot_count": len(self.plot_paths),
            "profile_report": str(self.profile_report_path) if self.profile_report_path else None,
            "business_insights": (
                str(self.business_insights_path) if self.business_insights_path else None
            ),
        }


class EDAAnalyzer:
    """Orchestrates complete exploratory data analysis pipeline."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: dict[str, Any],
        eda_cfg: dict[str, Any],
        validation_result: ValidationResult,
    ) -> None:
        self.df = df
        self.schema = schema
        self.cfg = eda_cfg
        self.validation_result = validation_result
        self.target_column = schema.get("target_column", "Attrition")
        self.numerical_columns = list(schema.get("numerical_columns", []))
        self.categorical_columns = list(schema.get("categorical_columns", []))

        self.output_dir = Path(str(eda_cfg["output_dir"]))
        self.plots_dir = Path(str(eda_cfg["plots_dir"]))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def run(self, generate_profile: bool = True) -> EDAResult:
        """Execute full EDA pipeline and return results."""
        logger.info("Starting EDA analysis -> %s", self.output_dir)
        plot_paths: list[Path] = []

        # Numerical distributions
        plot_paths.extend(
            plot_numerical_distributions(self.df, self.numerical_columns, self.plots_dir, self.cfg)
        )

        # Outlier analysis
        outlier_plots, outlier_summary = plot_outlier_boxplots(
            self.df, self.numerical_columns, self.plots_dir, self.cfg
        )
        plot_paths.extend(outlier_plots)

        # Correlations
        corr_plots, pearson_corr, _ = plot_correlation_heatmaps(
            self.df, self.numerical_columns, self.plots_dir, self.cfg
        )
        plot_paths.extend(corr_plots)

        # Pair plot on top correlated features
        top_features = select_top_correlated_features(
            pearson_corr,
            target_col=None,
            max_features=int(self.cfg.get("pairplot", {}).get("max_features", 6)),
        )
        plot_paths.extend(
            plot_pairplot(self.df, top_features, self.target_column, self.plots_dir, self.cfg)
        )

        # Class imbalance
        imbalance_stats = analyze_class_imbalance(
            self.df, self.target_column, self.plots_dir, self.cfg
        )

        # Categorical frequencies
        plot_paths.extend(
            plot_categorical_frequencies(
                self.df,
                self.categorical_columns,
                self.target_column,
                self.plots_dir,
                self.cfg,
            )
        )

        # YData profiling report
        profile_path: Path | None = None
        if generate_profile:
            profile_path = generate_profile_report(
                self.df,
                self.output_dir / "profile_report.html",
                self.cfg,
            )

        # Business insights
        insights_content = generate_business_insights(
            self.df,
            self.target_column,
            self.cfg,
            imbalance_stats,
            outlier_summary,
        )
        insights_path = save_business_insights(
            insights_content,
            self.output_dir / "business_insights.md",
        )

        # Summary JSON
        result = EDAResult(
            output_dir=self.output_dir,
            plots_dir=self.plots_dir,
            validation_result=self.validation_result,
            imbalance_stats=imbalance_stats,
            plot_paths=plot_paths,
            profile_report_path=profile_path,
            business_insights_path=insights_path,
        )

        summary_path = self.output_dir / "eda_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_summary_dict(), f, indent=2)
        result.summary_path = summary_path

        logger.info(
            "EDA complete: %d plots, summary at %s",
            len(plot_paths),
            summary_path,
        )
        return result
