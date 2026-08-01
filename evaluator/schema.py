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

# 슬라이딩 윈도우 추세 피처 (Day4, RL 시뮬레이션 중 실시간 계산 — rl_env/trend_features.py 참고)
# 최근 TREND_WINDOW_LAPS랩의 CLASS_POSITION/GAP_TO_LEADER_SEC에 대해
# 변화량(change=끝값-시작값)과 선형회귀 기울기(slope)를 계산해 붙인다.
TREND_WINDOW_LAPS: int = 5

TREND_FEATURE_COLUMNS: list[str] = [
    "pos_change_5lap",
    "gap_change_5lap",
    "pos_slope_5lap",
    "gap_slope_5lap",
]

# 팀원 B가 트렌드 피처까지 포함해 재학습한 모델(xgb_podium_model_tuned2.joblib) 입력 순서.
FEATURE_COLUMNS_TREND: list[str] = FEATURE_COLUMNS + TREND_FEATURE_COLUMNS

# 트랙 상태(세이프티카/FCY) 피처 — rl_env/podium_evaluator_trend_grid.json(V3, 2026-08-01 B 전달)
# 입력에 쓰임. 원본은 preprocessing/track_status.py가 만드는 TRACK_STATUS
# (GREEN/SAFETY_CAR/FULL_COURSE_YELLOW/FINISH) 컬럼 하나뿐인데, B가 이걸 7개 컬럼으로
# 인코딩해서 모델에 넣었다. 컬럼명이 헷갈리게 겹치는데, B에게 직접 확인한 매핑은 다음과 같다:
#   - TRACK_STATUS_SC == TRACK_STATUS_SAFETY_CAR (중복 컬럼, 같은 값)
#   - TRACK_STATUS_YELLOW == TRACK_STATUS_FULL_COURSE_YELLOW (중복 컬럼, 같은 값)
#   - TRACK_STATUS_CODE: 순서형 인코딩. GREEN=0, SAFETY_CAR=1, FULL_COURSE_YELLOW=2
#     (FINISH의 CODE 값은 B에게 확인 안 됨 — rl_env/pitstop_env.py::encode_track_status()에서
#     0으로 폴백하는 걸로 가정해뒀으니, 확인되면 여기 주석과 함께 고칠 것)
TRACK_STATUS_CODE_MAP: dict[str, int] = {"GREEN": 0, "SAFETY_CAR": 1, "FULL_COURSE_YELLOW": 2}

TRACK_STATUS_FEATURE_COLUMNS: list[str] = [
    "TRACK_STATUS_SC",
    "TRACK_STATUS_YELLOW",
    "TRACK_STATUS_CODE",
    "TRACK_STATUS_GREEN",
    "TRACK_STATUS_SAFETY_CAR",
    "TRACK_STATUS_FULL_COURSE_YELLOW",
    "TRACK_STATUS_FINISH",
]

FEATURE_COLUMNS_TREND_TRACK: list[str] = FEATURE_COLUMNS_TREND + TRACK_STATUS_FEATURE_COLUMNS

# 학습 라벨 — 최종 클래스 순위 3위 이내 여부 (bool)
TARGET_COLUMN: str = "PODIUM"

# 같은 레이스가 train/val에 안 섞이게 groupby할 때 쓰는 키 (README 참고)
GROUP_KEYS: list[str] = ["EVENT_ID", "NUMBER"]

# WEATHER_CATEGORY는 이미 정수 인코딩이라 원-핫/라벨인코딩 추가 처리가 필요 없음.
# (참고용) 인코딩 매핑: 0=DRY, 1=WET, 2=MIXED
WEATHER_CATEGORY_MAP: dict[int, str] = {0: "DRY", 1: "WET", 2: "MIXED"}

# 액션 마스킹 상수 — preprocessing/config.py 와 반드시 동일값 유지 (README 경고 참고)
# 2026-08-01: 34는 실제 스틴트 길이(2025 SPA 4대 평균 15~21랩)보다 너무 길어서
# 규칙 기반 에이전트가 사실상 항상 강제 피트로만 동작하는 문제가 있어 20으로 조정.
MAX_STINT_LAPS: int = 20
MIN_STINT_LAPS: int = 3
