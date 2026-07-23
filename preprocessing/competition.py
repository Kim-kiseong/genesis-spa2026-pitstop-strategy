"""
2-1-3. 경쟁 상황(순위 및 격차) 재구성 (기획서 명세 그대로 구현)

핵심 변수: ELAPSED (레이스 시작 후 누적 경과 시간)
처리 로직:
  1. 동일 ELAPSED 시간대(동일 랩 시점)에 다른 Hypercar 차량들의 위치를 역산해
     해당 랩 시점의 실시간 클래스 순위(CLASS_POSITION) 도출
  2. 선두 차량과의 시간 격차(GAP_TO_LEADER_SEC) 및 바로 앞 차량과의 격차(GAP_TO_AHEAD_SEC) 계산
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from time_utils import parse_time_str_to_seconds


def reconstruct_competition_state(laps: pd.DataFrame) -> pd.DataFrame:
    """
    laps: EVENT_ID, NUMBER, LAP_NUMBER, ELAPSED 컬럼 필요 (동일 클래스로 이미 필터링된 상태).
    각 랩 시점마다, 그 시점에서 "가장 최근에 완료한 랩"까지의 ELAPSED를 기준으로
    같은 이벤트의 모든 차량을 정렬해 순위/격차를 매긴다.
    """
    required = {"EVENT_ID", "NUMBER", "LAP_NUMBER", "ELAPSED"}
    missing = required - set(laps.columns)
    if missing:
        raise ValueError(f"reconstruct_competition_state: 필수 컬럼 누락 {missing}")

    laps = laps.copy()
    laps["ELAPSED_S"] = parse_time_str_to_seconds(laps["ELAPSED"])
    laps = laps.dropna(subset=["ELAPSED_S"])

    out_chunks = []
    for event_id, ev in laps.groupby("EVENT_ID", sort=False):
        ev = ev.sort_values("ELAPSED_S").reset_index(drop=True)
        # 각 (차량,랩) 완료 시점 스냅샷을 시간순으로 훑으며,
        # 그 시각까지 각 차량이 "가장 최근에 완료한 랩 수"로 순위를 매긴다 (레이스 진행률 기준).
        results = []
        # 차량별 최근 완료 랩 수 / 완료 시각 추적
        latest_lap = {}
        latest_time = {}
        snapshots = ev[["NUMBER", "LAP_NUMBER", "ELAPSED_S"]].itertuples(index=False)
        rows = list(snapshots)
        for number, lap_number, elapsed_s in rows:
            latest_lap[number] = lap_number
            latest_time[number] = elapsed_s
            # 현재 시점 기준 순위표 스냅샷: 완료 랩 수 내림차순, 동률이면 시간 오름차순(더 빨리 그 랩을 끝낸 차가 앞섬)
            standings = sorted(
                latest_time.items(),
                key=lambda kv: (-latest_lap[kv[0]], kv[1]),
            )
            positions = {num: pos + 1 for pos, (num, _t) in enumerate(standings)}
            leader_time = standings[0][1]
            leader_num = standings[0][0]
            # 앞 차량 시간
            pos_index = [n for n, _ in standings].index(number)
            ahead_time = standings[pos_index - 1][1] if pos_index > 0 else latest_time[number]

            results.append({
                "EVENT_ID": event_id,
                "NUMBER": number,
                "LAP_NUMBER": lap_number,
                "CLASS_POSITION": positions[number],
                "GAP_TO_LEADER_SEC": max(0.0, latest_time[number] - leader_time) if number != leader_num else 0.0,
                "GAP_TO_AHEAD_SEC": max(0.0, latest_time[number] - ahead_time),
            })
        out_chunks.append(pd.DataFrame(results))

    comp = pd.concat(out_chunks, ignore_index=True)
    merged = laps.merge(comp, on=["EVENT_ID", "NUMBER", "LAP_NUMBER"], how="left")
    return merged


def compute_lap_progress_ratio(laps: pd.DataFrame, total_laps_col: str = "TOTAL_RACE_LAPS") -> pd.DataFrame:
    """
    LAP_PROGRESS_RATIO = LAP_NUMBER / 해당 레이스의 예상 총 랩 수.
    total_laps_col이 없으면 이벤트 내 최댓값(LAP_NUMBER)으로 근사한다(레이스 종료 후 계산 시 적합).
    """
    laps = laps.copy()
    if total_laps_col not in laps.columns:
        max_laps = laps.groupby("EVENT_ID")["LAP_NUMBER"].transform("max")
    else:
        max_laps = laps[total_laps_col]
    laps["LAP_PROGRESS_RATIO"] = laps["LAP_NUMBER"] / max_laps.replace(0, np.nan)
    return laps
