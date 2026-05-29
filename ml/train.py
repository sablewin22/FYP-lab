import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


def train(X_train, y_train, X_test, y_test, classes, n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train.values if hasattr(X_train, 'values') else X_train, y_train)
    y_pred = model.predict(X_test.values if hasattr(X_test, 'values') else X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    Path(MODEL_PATH.parent).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Acurácia: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    print("\nConfusion Matrix:")
    print(cm)

    return model, acc


if __name__ == "__main__":
    from data.loader import get_dataframe
    from ml.preprocess import preprocess
    df = get_dataframe()
    X_train, X_test, y_train, y_test, classes = preprocess(df, fit=True)
    train(X_train, y_train, X_test, y_test, classes)
