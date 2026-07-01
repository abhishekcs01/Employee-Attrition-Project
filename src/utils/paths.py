"""Project path resolution utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathResolver:
    """Resolve project-relative paths from configuration."""

    project_root: Path
    data_raw: Path
    data_synthetic: Path
    data_processed: Path
    models: Path
    reports: Path
    outputs: Path

    @classmethod
    def from_config(cls, cfg: dict) -> PathResolver:
        """Build a PathResolver from a Hydra/OmegaConf paths section."""
        paths = cfg.get("paths", cfg)
        project_root = Path(str(paths["project_root"])).resolve()

        return cls(
            project_root=project_root,
            data_raw=project_root / "data" / "raw",
            data_synthetic=project_root / "data" / "synthetic",
            data_processed=project_root / "data" / "processed",
            models=project_root / "models",
            reports=project_root / "reports",
            outputs=project_root / "outputs",
        )

    def ensure_dirs(self) -> None:
        """Create all standard project directories if they do not exist."""
        for path in (
            self.data_raw,
            self.data_synthetic,
            self.data_processed,
            self.models,
            self.reports,
            self.outputs,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative: str) -> Path:
        """Resolve a path relative to the project root."""
        return (self.project_root / relative).resolve()
