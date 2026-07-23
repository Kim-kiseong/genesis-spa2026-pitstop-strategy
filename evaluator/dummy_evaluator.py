"""
evaluator/dummy_evaluator.py
=============================
Stage 1 (ML Scientist, 주윤서) 더미 평가기.

목적
----
실제 XGBClassifier 학습이 끝나기 전까지, 김기성(RL Engineer)이
MaskablePPO 보상 함수 배선을 지금 바로 진행할 수 있도록, 실제 모델과
**동일한 인터페이스**를 갖는 가짜 평가기를 제공한다.

컬럼명은 evaluator/schema.py 에서 가져온다 (하드코딩 없음). 이 이름들은
preprocessing/README.md 에 문서화된, 이미 실데이터(laps_train.parquet,
laps_val.parquet)로 검증 완료된 컬럼명이다.

인터페이스 계약 (진짜 모델로 교체될 때도 이 시그니처는 유지됨)
----------------------------------------------------------
- predict_proba(X: pd.DataFrame) -> np.ndarray, shape (n_samples,), 값 범위 [0, 1]
- from_parquet(path) -> parquet 을 읽어서 스키마 검증 후 확률 반환까지 한 번에

실제 모델이 준비되면 evaluator/xgb_evaluator.py 로 교체하고,
predict_proba / from_parquet 시그니처만 동일하게 맞추면 rl_env/ 쪽
코드는 한 줄도 안 바꿔도 된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluator.schema import FEATURE_COLUMNS, TARGET_COLUMN


class DummyPodiumEvaluator:
    """실제 XGBClassifier가 준비되기 전 사용하는 랜덤 확률 더미 평가기."""

    def __init__(self, feature_columns: list[str] | None = None, seed: int = 42):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self._rng = np.random.default_rng(seed)

    def _validate(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(
                f"[DummyPodiumEvaluator] 누락된 입력 컬럼: {missing}\n"
                f"evaluator/schema.py 의 FEATURE_COLUMNS와 laps_train.parquet의 "
                f"실제 컬럼이 일치하는지 확인하세요."
            )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """실제 학습은 하지 않음. 인터페이스 호환용 no-op."""
        self._validate(X)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """포디움(top-3) 확률을 랜덤하게 반환. shape: (n_samples,), 범위 [0, 1]."""
        self._validate(X)
        return self._rng.uniform(0.0, 1.0, size=len(X))

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @classmethod
    def from_parquet(cls, path: str, **kwargs) -> tuple["DummyPodiumEvaluator", np.ndarray]:
        """
        laps_train.parquet / laps_val.parquet 을 바로 넣어서 확률까지 뽑는
        진입점. 실제 모델로 바꿀 때도 이 함수 시그니처는 유지.
        """
        df = pd.read_parquet(path, columns=FEATURE_COLUMNS)
        evaluator = cls(**kwargs)
        probs = evaluator.predict_proba(df)
        return evaluator, probs


if __name__ == "__main__":
    import sys
    from pathlib import Path

    default_path = Path(__file__).resolve().parent.parent / "preprocessing" / "processed" / "laps_train.parquet"
    path = sys.argv[1] if len(sys.argv) > 1 else str(default_path)

    evaluator, probs = DummyPodiumEvaluator.from_parquet(path)
    print(f"입력 파일: {path}")
    print(f"입력 shape: ({len(probs)}, {len(FEATURE_COLUMNS)})")
    print(f"타겟 컬럼(참고용, 학습 안 함): {TARGET_COLUMN}")
    print("출력 확률 샘플:", np.round(probs[:5], 3))
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    print("스모크 테스트 통과: 실제 laps_train.parquet 로 predict_proba 정상 동작")
