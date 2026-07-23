"""
evaluator/train_xgboost.py
============================
주윤서(ML Scientist) 담당 - 정제된 데이터를 받아 포디움 확률을 예측하는
XGBoost 모델 학습 + SHAP 분석.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "LAP_PROGRESS_RATIO",
    "STINT_LAP",
    "CLASS_POSITION",
    "GAP_TO_LEADER_SEC",
    "GAP_TO_AHEAD_SEC",
    "WEATHER_CATEGORY",
    "TRACK_TEMP",
]
TARGET_COLUMN = "PODIUM"

THIS_DIR = Path(__file__).resolve().parent
TRAIN_PATH = THIS_DIR.parent / "preprocessing" / "processed" / "laps_train.parquet"
VAL_PATH = THIS_DIR.parent / "preprocessing" / "processed" / "laps_val.parquet"
MODEL_OUT_PATH = THIS_DIR / "xgb_podium_model.joblib"


def load_split(path: Path):
    df = pd.read_parquet(path, columns=FEATURE_COLUMNS + [TARGET_COLUMN])
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def main():
    print(f"train 로드: {TRAIN_PATH}")
    X_train, y_train = load_split(TRAIN_PATH)
    print(f"val 로드:   {VAL_PATH}")
    X_val, y_val = load_split(VAL_PATH)

    print(f"train shape: {X_train.shape}, 포디움 비율: {y_train.mean():.3f}")
    print(f"val shape:   {X_val.shape}, 포디움 비율: {y_val.mean():.3f}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"\nval AUC: {auc:.4f}")

    joblib.dump(model, MODEL_OUT_PATH)
    print(f"모델 저장: {MODEL_OUT_PATH}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\nSHAP 기준 피처 중요도 (평균 |SHAP|):")
    print(importance.to_string())

    return model, explainer, shap_values, X_val


if __name__ == "__main__":
    main()
