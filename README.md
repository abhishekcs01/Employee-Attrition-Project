# Employee Attrition Prediction

Machine learning case study predicting employee attrition from the IBM HR Analytics dataset. **Phase 1 (Baseline Model)** is the current deliverable — a notebook series from data understanding through baseline evaluation.

## Project Overview

| Stage | Notebook | Focus |
|-------|----------|-------|
| Overview | 00 | Introduction and roadmap |
| Data Understanding | 01 | Load, quality checks, EDA |
| Preprocessing | 02 | Cleaning, stratified split |
| Feature Engineering | 03 | Domain-informed HR features |
| Baseline Model | 04 | Logistic regression |
| Evaluation | 05 | Metrics, curves, recommendations |

## Current Milestone

**Phase 1 — Baseline Model** (complete): Notebooks `00`–`05`

Phases 2–5 (model comparison, synthetic data, SHAP, deployment) are documented in [PROJECT_GUIDE.md](PROJECT_GUIDE.md) and scaffolded under `src/` for future work.

## Repository Structure

```
Employee-Attrition Project/
├── notebooks/00–05          # Phase 1 deliverable
├── notebooks/_shared/       # Plotting theme, paths, I/O helpers
├── data/raw/                # IBM HR Analytics CSV
├── data/interim/            # Notebook artifacts (generated)
├── models/phase1/           # Baseline model (generated)
├── src/                     # Future pipeline (Phase 2+)
├── tests/
├── Makefile
├── PROJECT_GUIDE.md         # Full technical reference
└── pyproject.toml
```

## Installation

```bash
make setup
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install
```

Python 3.11+ required.

## Environment Setup

Activate the virtual environment before running commands:

```bash
source .venv/bin/activate
```

## Notebook Execution Order

1. `00_Project_Overview.ipynb`
2. `01_Data_Understanding_And_EDA.ipynb`
3. `02_Data_Preprocessing.ipynb`
4. `03_Feature_Engineering.ipynb`
5. `04_Baseline_Logistic_Regression.ipynb`
6. `05_Model_Evaluation.ipynb`

## How to Run

**Execute the full Phase 1 series:**

```bash
make phase1
```

**Launch JupyterLab:**

```bash
make notebook
```

**Run individual steps:**

```bash
make overview      # 00
make eda           # 01
make preprocess    # 02
make features      # 03
make baseline      # 04
make evaluate      # 05
```

Run from the **project root** with a fresh kernel: `Kernel → Restart & Run All`.

## Documentation

See **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** for project vision, architecture, design decisions, dataset details, and future roadmap.

## Future Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Baseline logistic regression (NB 00–05) | **Complete** |
| 2 | Model comparison (RF, XGBoost, LightGBM) | Planned |
| 3 | CTGAN synthetic data | Planned |
| 4 | SHAP explainability | Planned |
| 5 | FastAPI / Streamlit deployment | Planned |

## License

MIT — see [LICENSE](LICENSE).
