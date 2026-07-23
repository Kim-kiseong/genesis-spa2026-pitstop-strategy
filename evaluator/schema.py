"""
evaluator/schema.py
====================
preprocessing/README.md 의 "B(evaluator/)에게 넘기는 인터페이스" 절을
그대로 코드로 옮긴 것. laps_train.parquet / laps_val.parquet 에 이미
이 컬럼들이 실데이터로 존재함 (2026-07-23 확인, 8537행).

공리원이 이미 실데이터로 검증 완료했기 때문에, 여기 있는 이름은
"오늘 확정될 예정"이 아니라 "이미 확정되어 parquet에 박혀 있는" 이름임.
바뀔 경우 preprocessing/README.md 쪽 표가 먼저 바뀌고 여기가 따라감.
"""

from __future__ import annotations

# XGBoost 입력 피처 (기획서 상태 변수, preprocessing/README.md 표 그대로)
FEATURE_COLUMNS: list[str] = [
    "LAP_PROGRESS_RATIO",   # 레이스 진행률 (0~1)
    "STINT_LAP",            # 스틴트 내 경과 랩 수 (타이어/연료 마모 대리 변수)
    "CLASS_POSITION",       # 실시간 클래스 순위
    "GAP_TO_LEADER_SEC",    # 선두와의 격차(초)
    "GAP_TO_AHEAD_SEC",     # 바로 앞 차량과의 격차(초)
    "WEATHER_CATEGORY",     # 0=DRY, 1=WET, 2=MIXED (이미 숫자 인코딩됨)
    "TRACK_TEMP",           # 트랙 온도
]

# 학습 라벨 — 최종 클래스 순위 3위 이내 여부 (bool)
TARGET_COLUMN: str = "PODIUM"

# 같은 레이스가 train/val에 안 섞이게 groupby할 때 쓰는 키 (README 참고)
GROUP_KEYS: list[str] = ["EVENT_ID", "NUMBER"]

# WEATHER_CATEGORY는 이미 정수 인코딩이라 원-핫/라벨인코딩 추가 처리가 필요 없음.
# (참고용) 인코딩 매핑: 0=DRY, 1=WET, 2=MIXED
WEATHER_CATEGORY_MAP: dict[int, str] = {0: "DRY", 1: "WET", 2: "MIXED"}

# 액션 마스킹 상수 — preprocessing/config.py 와 반드시 동일값 유지 (README 경고 참고)
MAX_STINT_LAPS: int = 34
MIN_STINT_LAPS: int = 3
