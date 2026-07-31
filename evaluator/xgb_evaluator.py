"""
evaluator/xgb_evaluator.py
============================
Day 3 - 주윤서(ML Scientist)가 train_xgboost.py로 학습해 덤프한
xgb_podium_model.joblib을 감싸는 wrapper.

dummy_evaluator.DummyPodiumEvaluator와 동일한 인터페이스
(predict_proba / from_parquet)를 유지하므로, rl_env/ 쪽 코드는
"어떤 평가기를 쓰는지" 신경 쓸 필요 없이 그대로 동작한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluator.schema import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "xgb_podium_model.joblib"


class XGBPodiumEvaluator:
    """train_xgboost.py가 저장한 XGBClassifier(.joblib)를 감싸는 평가기."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, feature_columns: list[str] | None = None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"[XGBPodiumEvaluator] 모델 파일이 없습니다: {model_path}\n"
                f"evaluator/train_xgboost.py 를 먼저 실행해서 .joblib을 생성하세요."
            )
        # joblib은 모델을 실제로 로드할 때만 필요하므로 지연 임포트한다.
        # (joblib/xgboost가 아직 안 깔린 환경에서도 더미 평가기 폴백은 죽지 않게)
        import joblib

        self.model = joblib.load(model_path)
        self.feature_columns = feature_columns or FEATURE_COLUMNS

    def _validate(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"[XGBPodiumEvaluator] 누락된 입력 컬럼: {missing}\n"
                f"evaluator/schema.py 의 FEATURE_COLUMNS와 입력 DataFrame의 컬럼이 일치하는지 확인하세요."
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """포디움(top-3) 확률. shape: (n_samples,), 범위 [0, 1]."""
        self._validate(X)
        return self.model.predict_proba(X[self.feature_columns])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @classmethod
    def from_parquet(cls, path: str, model_path: str | Path = DEFAULT_MODEL_PATH, **kwargs):
        df = pd.read_parquet(path, columns=FEATURE_COLUMNS)
        evaluator = cls(model_path=model_path, **kwargs)
        probs = evaluator.predict_proba(df)
        return evaluator, probs


def load_evaluator(model_path: str | Path = DEFAULT_MODEL_PATH):
    """
    실제 XGBoost 모델(.joblib)이 있으면 그걸 로드하고,
    아직 B가 모델을 넘기기 전이라면 DummyPodiumEvaluator로 자동 폴백한다.
    (Day3 진행 중 B/C 작업이 서로 블로킹되지 않도록 하기 위함)
    """
    try:
        evaluator = XGBPodiumEvaluator(model_path=model_path)
        print(f"[load_evaluator] 실제 XGBoost 모델 로드 완료: {model_path}")
        return evaluator
    except (FileNotFoundError, ImportError) as e:
        from evaluator.dummy_evaluator import DummyPodiumEvaluator

        print(f"[load_evaluator] 경고: {e}\n-> DummyPodiumEvaluator로 폴백합니다.")
        return DummyPodiumEvaluator()


if __name__ == "__main__":
    import sys

    default_path = Path(__file__).resolve().parent.parent / "preprocessing" / "processed" / "laps_val.parquet"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default_path)

    evaluator = load_evaluator()
    df = pd.read_parquet(path, columns=FEATURE_COLUMNS)
    probs = evaluator.predict_proba(df)
    print(f"입력 파일: {path}")
    print(f"출력 확률 샘플: {np.round(probs[:5], 3)}")
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    print("스모크 테스트 통과: predict_proba 정상 동작")
