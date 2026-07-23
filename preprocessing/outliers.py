"""
Day 3. 이상치(Outlier) 제거.
스프린트 문서 꿀팁 3번: "레드플래그 같은 복잡한 예외는 파싱에 시간 쏟지 말고 드롭"을 반영.
FLAG_AT_FL의 RF(레드플래그)로 이미 정확히 식별되므로, 드롭은 한 줄이면 된다
(레드플래그 탐지 로직을 따로 만들 필요가 없었다는 게 지난 검증에서 나온 핵심 발견).
"""
from __future__ import annotations
import pandas as pd
from config import DROP_TRACK_STATUS, PIT_LOSS_OUTLIER_THRESHOLD_S


def drop_track_status_outliers(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """반환: (정상 랩, 드롭된 랩 — 케이스 스터디용으로 별도 보관)"""
    mask = laps["TRACK_STATUS"].isin(DROP_TRACK_STATUS)
    return laps[~mask].copy(), laps[mask].copy()


def flag_pit_loss_outliers(laps: pd.DataFrame) -> pd.DataFrame:
    """
    PIT_LOSS_S가 비정상적으로 큰 랩(사고/기계고장 등)에 OUTLIER_PIT_LOSS 플래그만 붙인다.
    행 자체를 드롭하지 않는 이유: stint_lap 시퀀스가 끊기면 뒤 랩들의 STINT_ID가 다 틀어짐.
    B(evaluator)가 학습 시 이 플래그로 걸러서 쓰면 됨.
    """
    laps = laps.copy()
    laps["OUTLIER_PIT_LOSS"] = laps["IS_PIT_LAP"] & (laps["PIT_LOSS_S"] > PIT_LOSS_OUTLIER_THRESHOLD_S)
    return laps
