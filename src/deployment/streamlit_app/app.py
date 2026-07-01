"""Streamlit dashboard for employee attrition prediction."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models/latest")


@st.cache_resource
def load_pipeline():
    """Load trained model pipeline."""
    return joblib.load(MODELS_DIR / "best_model.joblib")


@st.cache_data
def load_metadata() -> dict:
    """Load training metadata."""
    return json.loads((MODELS_DIR / "training_metadata.json").read_text(encoding="utf-8"))


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(page_title="Employee Attrition", layout="wide")
    st.title("Employee Attrition Prediction")

    if not (MODELS_DIR / "best_model.joblib").exists():
        st.error("No trained model found. Run `make train` first.")
        return

    pipeline = load_pipeline()
    metadata = load_metadata()
    raw_features = metadata.get("raw_feature_names", metadata.get("feature_names", []))
    threshold = float(metadata.get("classification_threshold", 0.5))

    st.sidebar.header("Employee Features")
    inputs: dict[str, object] = {}
    for feature in raw_features:
        inputs[feature] = st.sidebar.text_input(feature, value="")

    if st.sidebar.button("Predict"):
        df = pd.DataFrame([inputs])
        for col in df.columns:
            if df[col].dtype == object:
                with contextlib.suppress(ValueError):
                    df[col] = pd.to_numeric(df[col])

        prob = float(pipeline.predict_proba(df)[0, 1])
        positive = metadata.get("target_meta", {}).get("positive_class", "Yes")

        st.metric("Attrition Probability", f"{prob:.2%}")
        st.write("Prediction:", positive if prob >= threshold else "No")


if __name__ == "__main__":
    main()
