"""YData profiling report generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from ydata_profiling import ProfileReport

logger = logging.getLogger(__name__)


def generate_profile_report(
    df: pd.DataFrame,
    output_path: Path,
    cfg: dict[str, Any],
) -> Path:
    """Generate HTML profiling report using YData Profiling."""
    profiling_cfg = cfg.get("profiling", {})
    title = profiling_cfg.get("title", "EDA Profile Report")
    minimal = bool(profiling_cfg.get("minimal", False))
    explorative = bool(profiling_cfg.get("explorative", True))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating YData profiling report (this may take a minute)...")
    profile = ProfileReport(
        df,
        title=title,
        minimal=minimal,
        explorative=explorative,
        progress_bar=False,
    )
    profile.to_file(str(output_path))
    logger.info("Profile report saved to %s", output_path)
    return output_path
