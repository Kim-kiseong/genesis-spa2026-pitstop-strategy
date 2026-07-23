"""
FIA WEC Al Kamel Systems 원본 CSV 로더.
- Analysis by Lap: 세미콜론(;) 구분, 랩 단위 타이밍 데이터
- Weather Report: 세미콜론(;) 구분, 시간별 기상 데이터

실제 컬럼명은 연도별로 약간씩 다를 수 있으므로, 로드 직후
COLUMN_ALIASES로 표준 이름으로 통일한다. 새 파일에서 KeyError가 나면
COLUMN_ALIASES에 실제 원본 컬럼명을 추가하면 된다.

실데이터 검증(2026-07)으로 확인된 주의사항:
  - NUMBER는 반드시 문자열로 읽어야 함. int로 읽으면 "007"(Aston Martin) 같은
    3자리 번호의 앞자리 0이 사라져 "7"(Toyota)과 차량이 뒤섞인다.
  - ELAPSED/LAP_TIME/PIT_TIME은 'M:SS.mmm' 또는 'H:MM:SS.mmm' 문자열이라
    pd.to_numeric으로는 전부 NaN이 됨 -> time_utils.parse_time_str_to_seconds 사용.
  - 실제 아카이브 파일명은 고정 규칙을 안 따르는 경우가 많아(예:
    '21_SPA_Analysis_Race_Hour_6.csv', '21spa_Weather_Race_Hour_6.csv') 유연하게 탐색한다.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

from time_utils import parse_time_str_to_seconds

# 표준명 -> 원본에서 흔히 쓰이는 이름들
COLUMN_ALIASES = {
    "NUMBER": ["NUMBER", "CAR_NUMBER", "NUM"],
    "CLASS": ["CLASS", "CATEGORY"],
    "LAP_NUMBER": ["LAP_NUMBER", "LAP_NUM", "LAP"],
    "LAP_TIME": ["LAP_TIME", "LAPTIME"],
    "ELAPSED": ["ELAPSED", "ELAPSED_TIME", "TOTAL_TIME"],
    "HOUR": ["HOUR", "TIME_OF_DAY", "LOCAL_TIME"],
    "CROSSING_FINISH_LINE_IN_PIT": ["CROSSING_FINISH_LINE_IN_PIT", "PIT", "CROSSING_FINISH_LINE_IN_PIT_"],
    "PIT_TIME": ["PIT_TIME", "PIT_STOP_TIME"],
    "FLAG_AT_FL": ["FLAG_AT_FL", "FLAG", "TRACK_STATUS"],
    "DRIVER_NUMBER": ["DRIVER_NUMBER", "DRIVER_NAME", "DRIVER"],
    "TOP_SPEED": ["TOP_SPEED", "KPH"],
}

WEATHER_COLUMN_ALIASES = {
    "TIME_UTC_SECONDS": ["TIME_UTC_SECONDS", "TIME_UTC", "UTC_SECONDS"],
    "RAIN": ["RAIN", "RAINFALL"],
    "TRACK_TEMP": ["TRACK_TEMP", "TRACK_TEMPERATURE"],
    "AIR_TEMP": ["AIR_TEMP", "AIR_TEMPERATURE"],
}


def _standardize_columns(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    rename = {}
    upper_cols = {c.strip().upper(): c for c in df.columns}
    for std_name, candidates in alias_map.items():
        for cand in candidates:
            if cand in upper_cols:
                rename[upper_cols[cand]] = std_name
                break
    df = df.rename(columns=rename)
    return df


def load_lap_csv(path: str | Path, class_filter: str | None = None) -> pd.DataFrame:
    """Analysis by Lap CSV 1개 파일을 읽어 표준 컬럼명 DataFrame으로 반환."""
    path = Path(path)
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False, dtype={"NUMBER": str})
    df = _standardize_columns(df, COLUMN_ALIASES)
    if "NUMBER" in df.columns:
        df["NUMBER"] = df["NUMBER"].str.strip()

    if class_filter and "CLASS" in df.columns:
        df = df[df["CLASS"].astype(str).str.upper().str.contains(class_filter.upper(), na=False)]

    # CROSSING_FINISH_LINE_IN_PIT 마커 정규화: 'B' 또는 결측 -> bool
    if "CROSSING_FINISH_LINE_IN_PIT" in df.columns:
        df["IS_PIT_LAP"] = df["CROSSING_FINISH_LINE_IN_PIT"].astype(str).str.strip().str.upper().eq("B")
    else:
        df["IS_PIT_LAP"] = False

    # PIT_TIME은 'H:MM:SS.mmm' 문자열이며, B마커가 찍힌 랩이 아니라 그 "다음" 랩(피트 아웃랩)에
    # 기록된다. 실측 피트로스를 그대로 쓰기 위해 파싱 후 한 랩 당겨서(B마커 랩에) 붙인다.
    # 주의: IS_PIT_LAP은 B마커로만 판정한다 — PIT_TIME>0을 OR로 더하면 B마커 랩과 그 다음
    # 아웃랩이 둘 다 피트랩으로 잡혀 피트 횟수가 2배로 뻥튀기된다(실데이터 검증으로 발견).
    if "PIT_TIME" in df.columns:
        df = df.sort_values(["NUMBER", "LAP_NUMBER"]).reset_index(drop=True)
        df["PIT_TIME_S"] = parse_time_str_to_seconds(df["PIT_TIME"])
        df["PIT_LOSS_S_ACTUAL"] = df.groupby("NUMBER")["PIT_TIME_S"].shift(-1)
    else:
        df["PIT_LOSS_S_ACTUAL"] = float("nan")

    return df


def load_weather_csv(path: str | Path) -> pd.DataFrame:
    """Weather Report CSV 1개 파일을 읽어 표준 컬럼명 DataFrame으로 반환."""
    path = Path(path)
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    df = _standardize_columns(df, WEATHER_COLUMN_ALIASES)

    if "RAIN" in df.columns:
        # 파일마다 RAIN이 연속값(mm/h)이거나 0/1 플래그인 경우가 혼재 -> RAIN > 0 이 둘 다 처리
        df["RAIN"] = pd.to_numeric(df["RAIN"], errors="coerce").fillna(0)

    return df


def load_classification_csv(path: str | Path, class_filter: str | None = None) -> pd.DataFrame:
    """
    Classification by Category CSV (최종 순위표) 로더.
    XGBoost 라벨(포디움 진입 여부)을 만들기 위해 이벤트별 최종 클래스 순위가 필요하다.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False, dtype={"NUMBER": str})
    df = _standardize_columns(df, {**COLUMN_ALIASES, "FINAL_CLASS_POSITION": ["POSITION", "CLASS_POSITION", "RANK"]})
    if "NUMBER" in df.columns:
        df["NUMBER"] = df["NUMBER"].str.strip()
    if class_filter and "CLASS" in df.columns:
        df = df[df["CLASS"].astype(str).str.upper().str.contains(class_filter.upper(), na=False)]
    return df


def _find_raw_file(raw_dir: Path, year: int, circuit: str, keyword: str) -> Path | None:
    """
    실제 아카이브 파일명은 '{year}_{circuit}_lap.csv' 같은 고정 규칙을 안 따르는 경우가 많다
    (예: '21_SPA_Analysis_Race_Hour_6.csv', '21spa_Weather_Race_Hour_6.csv').
    2자리 연도 + 서킷명 + 키워드(analysis/weather/classification)가 모두 파일명에 포함되는
    파일을 대소문자 무시하고 찾는다. 못 찾으면 {year}_{circuit}_{keyword}.csv 고정 규칙도 시도.
    """
    yy = str(year)[2:]
    circuit_l = circuit.lower()
    for path in raw_dir.glob("*.csv"):
        name_l = path.name.lower()
        if yy in name_l and circuit_l in name_l and keyword.lower() in name_l:
            return path
    fixed = raw_dir / f"{year}_{circuit}_{keyword.lower()}.csv"
    return fixed if fixed.exists() else None


def load_event(event: dict, raw_dir: Path, class_filter: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    event = {"year": 2023, "circuit": "SPA"} 형태.
    classification 파일이 없으면 None을 반환(호출부에서 자체 순위로 근사 처리).
    """
    year, circuit = event["year"], event["circuit"]
    lap_path = _find_raw_file(raw_dir, year, circuit, "analysis") or _find_raw_file(raw_dir, year, circuit, "lap")
    weather_path = _find_raw_file(raw_dir, year, circuit, "weather")
    classification_path = _find_raw_file(raw_dir, year, circuit, "classification")

    if lap_path is None:
        raise FileNotFoundError(f"랩 데이터 없음: raw_dir={raw_dir}, event={year}_{circuit}")
    if weather_path is None:
        raise FileNotFoundError(f"날씨 데이터 없음: raw_dir={raw_dir}, event={year}_{circuit}")

    laps = load_lap_csv(lap_path, class_filter=class_filter)
    weather = load_weather_csv(weather_path)
    classification = load_classification_csv(classification_path, class_filter=class_filter) if classification_path is not None else None

    laps["EVENT_YEAR"] = year
    laps["EVENT_CIRCUIT"] = circuit
    laps["EVENT_ID"] = f"{year}_{circuit}"

    return laps, weather, classification
