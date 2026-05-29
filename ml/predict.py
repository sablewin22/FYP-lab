import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from ml.preprocess import ALL_FEATURES, NUM_FEATURES, PREPROCESSOR_PATH

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


def _load_artifacts():
    model = joblib.load(MODEL_PATH)
    pre = joblib.load(PREPROCESSOR_PATH)
    return model, pre["scaler"], pre["label_encoder"], pre["classes"]


def predict(input_dict: dict):
    model, scaler, le, classes = _load_artifacts()
    df = pd.DataFrame([input_dict])[ALL_FEATURES]
    df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])

    pred_idx = model.predict(df.values)[0]
    pred_class = classes[pred_idx]
    probs = model.predict_proba(df.values)[0]

    probabilities = {cls: float(prob) for cls, prob in zip(classes, probs)}

    importances = model.feature_importances_
    top5_idx = np.argsort(importances)[-5:][::-1]
    top5 = {ALL_FEATURES[i]: float(importances[i]) for i in top5_idx}

    return pred_class, probabilities, top5
