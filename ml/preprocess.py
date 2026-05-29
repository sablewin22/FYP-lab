import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

CAT_FEATURES = ["platform_cat", "region_cat", "language_cat", "category_cat",
                "traffic_source_cat", "creator_tier_cat"]
NUM_FEATURES = ["title_len", "text_richness", "like_rate_log", "comment_rate_log",
                "share_rate_log", "views_per_day"]
BIN_FEATURES = ["weekend_hashtag_boost"]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES + BIN_FEATURES
TARGET = "trend_label"

PREPROCESSOR_PATH = Path(__file__).resolve().parent.parent / "models" / "preprocessor.pkl"


def preprocess(df: pd.DataFrame, fit: bool = True, test_size: float = 0.2, random_state: int = 42):
    df = df.dropna(subset=ALL_FEATURES + [TARGET]).copy()
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    if fit:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        scaler = StandardScaler()
        X[NUM_FEATURES] = scaler.fit_transform(X[NUM_FEATURES])
        to_save = {"scaler": scaler, "label_encoder": le, "classes": le.classes_.tolist()}
        Path(PREPROCESSOR_PATH.parent).mkdir(parents=True, exist_ok=True)
        joblib.dump(to_save, PREPROCESSOR_PATH)
    else:
        loaded = joblib.load(PREPROCESSOR_PATH)
        le = loaded["label_encoder"]
        scaler = loaded["scaler"]
        y_enc = le.transform(y)
        X[NUM_FEATURES] = scaler.transform(X[NUM_FEATURES])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )
    return X_train, X_test, y_train, y_test, le.classes_


if __name__ == "__main__":
    from data.loader import get_dataframe
    df = get_dataframe()
    X_train, X_test, y_train, y_test, classes = preprocess(df, fit=True)
    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
    print(f"Classes: {classes}")
