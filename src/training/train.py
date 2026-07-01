"""CLI entry point for model training."""

from src.training.run import main, run_training_pipeline

__all__ = ["main", "run_training_pipeline"]


if __name__ == "__main__":
    main()
