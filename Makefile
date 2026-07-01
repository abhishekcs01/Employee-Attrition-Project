.PHONY: setup install lint format test notebook \
	overview eda preprocess features baseline evaluate phase1 clean help

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest
NB_TIMEOUT ?= 600

PYTHONPATH := "$(CURDIR)"
NBEXEC = mkdir -p .ipython .jupyter && \
	PYTHONPATH=$(PYTHONPATH) IPYTHONDIR=$$(pwd)/.ipython JUPYTER_DATA_DIR=$$(pwd)/.jupyter \
	$(PY) -m jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.kernel_name=employee-attrition \
		--ExecutePreprocessor.timeout=$(NB_TIMEOUT)

help:
	@echo "Employee Attrition ML Project — Phase 1"
	@echo ""
	@echo "  make setup            Create virtual environment"
	@echo "  make install          Install package with dev dependencies"
	@echo "  make notebook         Launch JupyterLab in notebooks/"
	@echo "  make overview         Execute Notebook 00 (project overview)"
	@echo "  make eda              Execute Notebook 01 (data understanding & EDA)"
	@echo "  make preprocess       Execute Notebook 02 (preprocessing & split)"
	@echo "  make features         Execute Notebook 03 (feature engineering)"
	@echo "  make baseline         Execute Notebook 04 (baseline logistic regression)"
	@echo "  make evaluate         Execute Notebook 05 (model evaluation)"
	@echo "  make phase1           Execute all Phase 1 notebooks (00-05)"
	@echo "  make lint             Run ruff linter"
	@echo "  make test             Run pytest with coverage"
	@echo "  make clean            Remove caches and generated outputs"

setup:
	$(PYTHON) -m venv $(VENV)
	@echo "Run: source $(VENV)/bin/activate && make install"

install:
	$(PIP) install -e ".[dev]"
	mkdir -p .jupyter/kernels/employee-attrition
	printf '%s\n' '{' \
		'  "argv": [' \
		'    "$(CURDIR)/$(VENV)/bin/python",' \
		'    "-m",' \
		'    "ipykernel_launcher",' \
		'    "-f",' \
		'    "{connection_file}"' \
		'  ],' \
		'  "display_name": "Python 3 (employee-attrition)",' \
		'  "language": "python"' \
		'}' > .jupyter/kernels/employee-attrition/kernel.json

lint:
	$(RUFF) check src tests notebooks/_shared

format:
	$(VENV)/bin/black src tests notebooks/_shared
	$(RUFF) check --fix src tests notebooks/_shared

test:
	$(PYTEST) tests/ -v --cov=src --cov-report=term-missing

notebook:
	$(PY) -m jupyter lab notebooks/

overview:
	$(NBEXEC) notebooks/00_Project_Overview.ipynb \
		--output 00_Project_Overview.ipynb

eda:
	$(NBEXEC) notebooks/01_Data_Understanding_And_EDA.ipynb \
		--output 01_Data_Understanding_And_EDA.ipynb

preprocess:
	$(NBEXEC) notebooks/02_Data_Preprocessing.ipynb \
		--output 02_Data_Preprocessing.ipynb

features:
	$(NBEXEC) notebooks/03_Feature_Engineering.ipynb \
		--output 03_Feature_Engineering.ipynb

baseline:
	$(NBEXEC) notebooks/04_Baseline_Logistic_Regression.ipynb \
		--output 04_Baseline_Logistic_Regression.ipynb

evaluate:
	$(NBEXEC) notebooks/05_Model_Evaluation.ipynb \
		--output 05_Model_Evaluation.ipynb

phase1: overview eda preprocess features baseline evaluate

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf catboost_info outputs/mlruns .coverage .ipython 2>/dev/null || true
	rm -rf data/interim/*.csv models/phase1/*.joblib 2>/dev/null || true
