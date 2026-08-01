"""
evaluator/xgb_evaluator.py
============================
Day 3 - 주윤서(ML Scientist)가 train_xgboost.py로 학습해 덤프한 모델을 감싸는 wrapper.

두 가지 저장 포맷을 지원한다:
  - 네이티브 포맷(.json/.ubj, model.save_model()) — 우선 사용. XGBoost가 버전 간
    호환을 공식 보장하는 포맷이라 pickle 버전 스큐 문제(예: 서로 xgboost 3.3.0으로
    버전이 같은데도 .joblib이 "input stream corrupted"로 깨졌던 사례)에서 자유롭다.
  - .joblib(pickle) — 레거시 경로. 위 문제가 재현되면 이쪽은 계속 실패할 수 있음.

컬럼 순서 주의: B가 학습에 쓴 컬럼 순서가 evaluator/schema.py::FEATURE_COLUMNS 순서와
다를 수 있다(실제로 달랐음 — 둘 다 7개 컬럼이지만 순서가 다름). XGBoost는 순서가 다르면
조용히 틀린 예측을 내지 않고 feature_names mismatch 에러를 던지므로, 여기서는 모델이
기억하는 실제 학습 순서(model.feature_names_in_)를 신뢰 소스로 삼아 입력을 재정렬한다.

dummy_evaluator.DummyPodiumEvaluator와 동일한 인터페이스
(predict_proba / from_parquet)를 유지하므로, rl_env/ 쪽 코드는
"어떤 평가기를 쓰는지" 신경 쓸 필요 없이 그대로 동작한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluator.schema import FEATURE_COLUMNS

MODEL_DIR = Path(__file__).resolve().parent

# 우선순위: 네이티브 포맷 먼저(B가 실제로 넘긴 파일명 포함), .joblib은 마지막 폴백.
CANDIDATE_MODEL_PATHS: list[Path] = [
    MODEL_DIR / "podium_evaluator.json",       # B가 실제로 넘긴 네이티브 포맷 파일명
    MODEL_DIR / "xgb_podium_model.ubj",
    MODEL_DIR / "xgb_podium_model.json",
    MODEL_DIR / "xgb_podium_model.joblib",     # 레거시 pickle 폴백
]
DEFAULT_MODEL_PATH = CANDIDATE_MODEL_PATHS[0]

NATIVE_SUFFIXES = {".json", ".ubj"}


def find_model_path(candidates: list[Path] | None = None) -> Path | None:
    """CANDIDATE_MODEL_PATHS 중 실제로 존재하는 첫 파일을 반환. 없으면 None."""
    for path in candidates or CANDIDATE_MODEL_PATHS:
        if path.exists():
            return path
    return None


class XGBPodiumEvaluator:
    """train_xgboost.py가 저장한 모델(.json/.ubj 네이티브 또는 .joblib)을 감싸는 평가기."""

    def __init__(self, model_path: str | Path | None = None, feature_columns: list[str] | None = None):
        if model_path is None:
            model_path = find_model_path()
            if model_path is None:
                searched = ", ".join(str(p) for p in CANDIDATE_MODEL_PATHS)
                raise FileNotFoundError(
                    f"[XGBPodiumEvaluator] 모델 파일이 없습니다. 다음 경로들을 찾아봤습니다: {searched}\n"
                    f"evaluator/train_xgboost.py 를 먼저 실행해서 모델을 생성하세요."
                )
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"[XGBPodiumEvaluator] 모델 파일이 없습니다: {model_path}")

        if model_path.suffix in NATIVE_SUFFIXES:
            # 네이티브 포맷: XGBoost 자체 스키마로 로드(버전 간 호환 보장, pickle 미사용).
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(str(model_path))
            self.model = model
        else:
            # 레거시 .joblib(pickle) 경로. joblib은 실제로 로드할 때만 필요하므로 지연 임포트.
            # (joblib/xgboost가 아직 안 깔린 환경에서도 더미 평가기 폴백은 죽지 않게)
            import joblib

            self.model = joblib.load(model_path)

        self.model_path = model_path
        self.feature_columns = feature_columns or FEATURE_COLUMNS

        # 모델이 실제로 학습된 컬럼 순서를 기억하고 있으면 그걸 신뢰 소스로 사용.
        # (schema.py의 FEATURE_COLUMNS 순서와 다를 수 있음 — 반드시 이 순서로 predict해야 함)
        self.model_feature_order = list(getattr(self.model, "feature_names_in_", self.feature_columns))

    def _validate(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.model_feature_order if c not in X.columns]
        if missing:
            raise ValueError(
                f"[XGBPodiumEvaluator] 누락된 입력 컬럼: {missing}\n"
                f"evaluator/schema.py 의 FEATURE_COLUMNS와 입력 DataFrame의 컬럼이 일치하는지 확인하세요."
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """포디움(top-3) 확률. shape: (n_samples,), 범위 [0, 1]."""
        self._validate(X)
        # 모델이 학습된 순서로 재정렬해서 넣는다 — 순서가 다르면 XGBoost가
        # feature_names mismatch 에러를 던지거나(신버전) 조용히 틀린 예측을 낼 수 있다.
        return self.model.predict_proba(X[self.model_feature_order])[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @classmethod
    def from_parquet(cls, path: str, model_path: str | Path | None = None, **kwargs):
        df = pd.read_parquet(path, columns=FEATURE_COLUMNS)
        evaluator = cls(model_path=model_path, **kwargs)
        probs = evaluator.predict_proba(df)
        return evaluator, probs


def load_evaluator(model_path: str | Path | None = None):
    """
    실제 XGBoost 모델(네이티브 .json/.ubj 우선, 없으면 .joblib)이 있으면 그걸 로드하고,
    아직 B가 모델을 넘기기 전이라면 DummyPodiumEvaluator로 자동 폴백한다.
    (Day3 진행 중 B/C 작업이 서로 블로킹되지 않도록 하기 위함)
    """
    try:
        evaluator = XGBPodiumEvaluator(model_path=model_path)
        print(f"[load_evaluator] 실제 XGBoost 모델 로드 완료: {evaluator.model_path}")
        print(f"[load_evaluator] 학습 시 컬럼 순서: {evaluator.model_feature_order}")
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
