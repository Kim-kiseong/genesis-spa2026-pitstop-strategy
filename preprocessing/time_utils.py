"""
Al Kamel CSV의 ELAPSED/LAP_TIME/PIT_TIME 컬럼은 'M:SS.mmm' 또는 'H:MM:SS.mmm' 문자열이다.
pd.to_numeric은 전부 NaN을 반환하고, pd.to_timedelta는 'M:SS.mmm'(2파트, 시 생략)을 못 읽는다.
두 포맷 다 처리하는 전용 파서를 여기 한 곳에 모아둔다 (data/*.py 여러 곳에서 공유).
"""
from __future__ import annotations
import pandas as pd


def parse_time_str_to_seconds(series: pd.Series) -> pd.Series:
    def _parse_one(s):
        if pd.isna(s):
            return float("nan")
        parts = str(s).strip().split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return float("nan")
        if len(parts) == 2:
            m, sec = parts
            return m * 60 + sec
        elif len(parts) == 3:
            h, m, sec = parts
            return h * 3600 + m * 60 + sec
        return float("nan")

    return series.map(_parse_one)
