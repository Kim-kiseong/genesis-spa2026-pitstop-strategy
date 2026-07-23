"""
트랙 상태 판별.

- 2021년 이후 데이터에는 공식 FLAG_AT_FL 컬럼이 존재 -> 이를 1차 소스로 사용.
- 2021 데이터처럼 이 컬럼이 없는 경우에만 휴리스틱으로 폴백(자리표시자, GREEN 고정).

실데이터 검증(2026-07)으로 해결됨: 기획서 3-2절 피드백 요청사항 1(레드플래그 탐지 방법론)은
FLAG_AT_FL 컬럼 안에 RF(레드플래그) 코드가 이미 존재해서 별도 로직이 필요 없었다.
당초 FF를 레드플래그로 추정했던 것이 오류였음 — FF는 체커기(레이스 종료 표시)이고,
RF가 실제 레드플래그다. RF는 2022_SPA, 2024_SPA 두 레이스에서만 등장한다(config.FLAG_VALUES 참고).
detect_red_flag_candidates()는 더 이상 필요하지 않지만, 혹시 FLAG_AT_FL이 없는 연도(2021)에서
레드플래그 여부를 참고용으로 추정하고 싶을 때 쓸 수 있도록 남겨둔다.
"""
from __future__ import annotations
import pandas as pd
from config import FLAG_VALUES


def apply_flag_at_fl(laps: pd.DataFrame) -> pd.DataFrame:
    laps = laps.copy()
    if "FLAG_AT_FL" in laps.columns and laps["FLAG_AT_FL"].notna().any():
        laps["TRACK_STATUS"] = laps["FLAG_AT_FL"].map(FLAG_VALUES).fillna("UNKNOWN")
        laps["TRACK_STATUS_SOURCE"] = "FLAG_AT_FL"
    else:
        laps["TRACK_STATUS"] = heuristic_track_status(laps)
        laps["TRACK_STATUS_SOURCE"] = "HEURISTIC_FALLBACK"
    return laps


def heuristic_track_status(laps: pd.DataFrame) -> pd.Series:
    """
    2021년처럼 FLAG_AT_FL이 없는 파일용 폴백. GREEN 고정 자리표시자.
    2021년 레드플래그/세이프티카 여부가 꼭 필요하면 detect_red_flag_candidates()로
    ELAPSED 총 소요시간 이상치를 참고 신호로만 활용 (다른 연도는 FLAG_AT_FL로 충분해 불필요).
    """
    return pd.Series("GREEN", index=laps.index)


def detect_red_flag_candidates(laps: pd.DataFrame, expected_race_seconds: float | None = None) -> pd.DataFrame:
    """
    2021년(FLAG_AT_FL 없음) 한정 참고용 보조 신호. 다른 연도는 FLAG_AT_FL의 RF로 충분하므로 불필요.
    레이스 총 소요시간이 예정보다 크게 늘어난 이벤트를 레드플래그 의심으로 표시하는 러프한 근사.
    """
    laps = laps.copy()
    laps["RED_FLAG_SUSPECTED"] = False
    if expected_race_seconds is None:
        return laps

    for event_id, ev in laps.groupby("EVENT_ID"):
        actual_duration = ev["ELAPSED_S"].max() if "ELAPSED_S" in ev.columns else None
        if actual_duration is None:
            continue
        overrun_ratio = actual_duration / expected_race_seconds
        if overrun_ratio > 1.15:  # 15% 이상 지연 -> 레드플래그 의심 (임시 임계값, 검증 필요)
            laps.loc[laps["EVENT_ID"] == event_id, "RED_FLAG_SUSPECTED"] = True

    return laps
