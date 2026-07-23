"""
2-1-2. 날씨 및 외부 환경 데이터 결합 (기획서 명세 그대로 구현)

핵심 변수: Al Kamel 날씨 데이터의 TIME_UTC_SECONDS, RAIN, TRACK_TEMP
처리 로직:
  1. 랩 데이터의 현지 시각(HOUR)에서 서머타임 오프셋(예: 스파 -2시간)을 차감해 UTC 변환
  2. RAIN 플래그 기준 DRY(0)/WET(1) 라벨 부여, 강수 전환 시점 전후 15분은 MIXED
  3. pd.merge_asof로 시간 오차 5분 이내에서 랩 데이터와 날씨 데이터 결합

주의(팀 명세서 예외 사항, 반드시 반영):
  - 23_SPA: RAIN 플래그가 불완전 -> WET/MIXED 구간 수동 라벨링 필요 (아래 MANUAL_WEATHER_OVERRIDES)
  - 24_SPA: 레드플래그로 인한 스틴트 길이 왜곡 있음 (날씨 자체와는 무관하나 같은 이벤트이므로 주석 처리)
  - 24_IMOLA: 비 blip 구간에 센서 결측 있음 -> 랩타임 변화와 교차검증 권장
"""
from __future__ import annotations
import pandas as pd
from config import (
    CIRCUIT_UTC_OFFSET_H, WEATHER_MERGE_TOLERANCE_MIN, MIXED_WINDOW_MIN,
    MANUAL_WEATHER_OVERRIDE_EVENTS,
)

WEATHER_DRY, WEATHER_WET, WEATHER_MIXED = 0, 1, 2

# 이벤트별 수동 보정 구간(랩 단위): {"EVENT_ID": [(start_utc_sec, end_utc_sec, label), ...]}
# 이벤트 "전체"를 강제 라벨링하려면 config.MANUAL_WEATHER_OVERRIDE_EVENTS를 쓴다(23_SPA가 그 경우).
# 여기는 "레이스 중 일부 구간만" 보정이 필요할 때 쓰는 세분화된 훅.
MANUAL_WEATHER_OVERRIDES: dict[str, list[tuple[float, float, int]]] = {
    # "2024_IMOLA": [(12345.0, 15678.0, WEATHER_MIXED)],  # 예: 랩타임 교차검증 후 확정되면 채우기
}


def _hour_to_utc_seconds(hour_col: pd.Series, circuit: str) -> pd.Series:
    """HOUR(현지 시각, HH:MM:SS 또는 초 단위)를 UTC 경과초(당일 자정 기준)로 변환."""
    offset_h = CIRCUIT_UTC_OFFSET_H.get(circuit, 0)
    if pd.api.types.is_numeric_dtype(hour_col):
        seconds = hour_col.astype(float)
    else:
        td = pd.to_timedelta(hour_col.astype(str), errors="coerce")
        seconds = td.dt.total_seconds()
    return seconds - offset_h * 3600


def _label_weather(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.copy()
    # weather의 TIME_UTC_SECONDS는 유닉스 epoch(1970년 기준 절대초)라서, HOUR 기반으로 계산한
    # '당일 자정 UTC 기준 경과초'와 스케일이 전혀 다르다(10^9 vs 10^4~10^5). 레이스가 UTC 자정을
    # 넘기지 않는다는 전제(모두 유럽 주간 6~8시간 레이스라 안전)하에 86400 나머지로 스케일을 맞춘다.
    weather["TIME_UTC_SECONDS"] = pd.to_numeric(weather["TIME_UTC_SECONDS"], errors="coerce").astype(float) % 86400
    weather = weather.sort_values("TIME_UTC_SECONDS")

    is_rain = weather["RAIN"] > 0
    weather["WEATHER_CATEGORY"] = is_rain.astype(int)  # 0=DRY, 1=WET 우선 부여

    # 강수 전환 시점(0->1 또는 1->0) 탐지 후 전후 MIXED_WINDOW_MIN 마킹
    transitions = weather.index[is_rain != is_rain.shift(1).fillna(is_rain.iloc[0])]
    window_s = MIXED_WINDOW_MIN * 60
    for idx in transitions:
        t0 = weather.loc[idx, "TIME_UTC_SECONDS"]
        mask = (weather["TIME_UTC_SECONDS"] >= t0 - window_s) & (weather["TIME_UTC_SECONDS"] <= t0 + window_s)
        weather.loc[mask, "WEATHER_CATEGORY"] = WEATHER_MIXED

    return weather


def _apply_manual_overrides(laps: pd.DataFrame, event_id: str) -> pd.DataFrame:
    if event_id in MANUAL_WEATHER_OVERRIDE_EVENTS:
        # 이벤트 전체 강제 라벨링 (예: 23_SPA — RAIN=0인데 노면은 젖어있던 예외 케이스)
        laps = laps.copy()
        laps["WEATHER_CATEGORY"] = MANUAL_WEATHER_OVERRIDE_EVENTS[event_id]
        return laps

    overrides = MANUAL_WEATHER_OVERRIDES.get(event_id)
    if not overrides:
        return laps
    laps = laps.copy()
    for start, end, label in overrides:
        mask = (laps["LAP_TIME_UTC_SECONDS"] >= start) & (laps["LAP_TIME_UTC_SECONDS"] <= end)
        laps.loc[mask, "WEATHER_CATEGORY"] = label
    return laps


def merge_weather(laps: pd.DataFrame, weather: pd.DataFrame, circuit: str, event_id: str) -> pd.DataFrame:
    laps = laps.copy()
    laps["LAP_TIME_UTC_SECONDS"] = _hour_to_utc_seconds(laps["HOUR"], circuit)
    laps["LAP_TIME_UTC_SECONDS"] = laps["LAP_TIME_UTC_SECONDS"].astype(float)

    weather = _label_weather(weather)

    laps = laps.sort_values("LAP_TIME_UTC_SECONDS")
    weather = weather.sort_values("TIME_UTC_SECONDS")

    merged = pd.merge_asof(
        laps,
        weather[["TIME_UTC_SECONDS", "RAIN", "TRACK_TEMP", "WEATHER_CATEGORY"]],
        left_on="LAP_TIME_UTC_SECONDS",
        right_on="TIME_UTC_SECONDS",
        direction="nearest",
        tolerance=WEATHER_MERGE_TOLERANCE_MIN * 60,
    )

    merged = _apply_manual_overrides(merged, event_id)
    return merged
