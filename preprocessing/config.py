"""
preprocessing/ 전용 설정 — 팀원 A(Data Engineer) 담당 범위.
B(evaluator/), C(rl_env/)는 이 파일을 직접 import하지 않고, laps_train.parquet /
laps_val.parquet의 컬럼 인터페이스(README.md 참고)만 보고 작업하면 됩니다.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RAW_DIR = ROOT_DIR / "raw_data"
PROCESSED_DIR = ROOT_DIR / "processed"

for d in (RAW_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

CLASS_FILTER = "HYPERCAR"

# Day 3: 연도별 train/val 스플릿 기준.
# 확정된 데이터 범위(2026-07 기준, 더 이상 늘어나지 않음): 21·22·23 SPA, 24 SPA/IMOLA,
# 25 SPA/IMOLA — 총 7개 이벤트. 21~23 IMOLA는 원본 아카이브에 제공되지 않아 학습에서 제외.
TRAIN_EVENTS = [
    {"year": 2021, "circuit": "SPA"},
    {"year": 2022, "circuit": "SPA"},
    {"year": 2023, "circuit": "SPA"},
    {"year": 2024, "circuit": "SPA"},
    {"year": 2024, "circuit": "IMOLA"},
]
VAL_EVENTS = [
    {"year": 2025, "circuit": "SPA"},
    {"year": 2025, "circuit": "IMOLA"},
]

# ── 서킷별 UTC 오프셋 (CEST=UTC+2, 부호 규칙: UTC = 현지시각 - 오프셋) ──
CIRCUIT_UTC_OFFSET_H = {"SPA": 2, "IMOLA": 2}

# ── 스틴트 임계값 (C의 액션 마스킹과 반드시 동일 값으로 맞출 것) ──
# 2026-08-01: 34는 실제 스틴트 길이(2025 SPA 4대 평균 15~21랩)보다 너무 길어서
# 규칙 기반 에이전트가 사실상 항상 강제 피트로만 동작하는 문제가 있어 20으로 조정.
MAX_STINT_LAPS = 20
MIN_STINT_LAPS = 3
PIT_LOSS_DEFAULT_S = 30.0

# ── 날씨 라벨링 ──
WEATHER_MERGE_TOLERANCE_MIN = 5
MIXED_WINDOW_MIN = 15
MANUAL_WEATHER_OVERRIDE_EVENTS = {"2023_SPA": 1}  # RAIN=0인데 실제 WET이었던 예외

# ── 트랙 상태 (RF=레드플래그, FF=체커기 — 실데이터로 검증된 정답) ──
FLAG_VALUES = {"GF": "GREEN", "SF": "SAFETY_CAR", "FCY": "FULL_COURSE_YELLOW", "RF": "RED_FLAG", "FF": "FINISH"}

# ── Day 3 이상치 제거 기준 ──
# 스프린트 문서 꿀팁 3번("이상치는 과감히 드롭") 반영: 레드플래그 구간 랩은 학습에서 제외.
# FCY/세이프티카는 유지(발생 빈도가 높아서 이것까지 빼면 데이터가 너무 줄어듦 — 필요시 조정).
DROP_TRACK_STATUS = ["RED_FLAG"]
# 피트로스가 이 값보다 크면 이상치로 간주해 학습에서 제외(사고/기계고장 등). 케이스 스터디용으로는 별도 보관.
PIT_LOSS_OUTLIER_THRESHOLD_S = 300.0

PODIUM_RANK_THRESHOLD = 3
