"""Tests for FastAPI deployment."""

from __future__ import annotations

import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.deployment.api.main import app, load_artifacts


@pytest.fixture
def model_artifacts(tmp_path):
    """Create minimal model artifacts for API tests."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    pipeline = Pipeline(
        [
            ("preprocess", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    )
    import numpy as np

    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y = np.array([0, 1, 0])
    pipeline.fit(x, y)

    joblib.dump(pipeline, models_dir / "best_model.joblib")
    metadata = {
        "best_model_name": "logistic_regression",
        "raw_feature_names": ["feat_a", "feat_b"],
        "target_meta": {"positive_class": "Yes", "negative_class": "No"},
        "classification_threshold": 0.5,
    }
    (models_dir / "training_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return models_dir


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint(model_artifacts):
    load_artifacts(model_artifacts)
    client = TestClient(app)
    response = client.post(
        "/predict",
        json={"features": {"feat_a": 1.5, "feat_b": 2.5}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "attrition_probability" in data
    assert data["attrition_prediction"] in {"Yes", "No"}
