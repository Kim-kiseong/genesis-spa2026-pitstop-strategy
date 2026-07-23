"""
2-1-1. Stint 및 랩 타임라인 복원 (기획서 명세 그대로 구현)

핵심 변수: LAP_NUMBER, CROSSING_FINISH_LINE_IN_PIT(마커: B), PIT_TIME
처리 로직:
  1. 피트레인 진입 마커(B)와 PIT_TIME이 발생한 랩을 '피트스탑 랩'으로 규정 (loaders.load_lap_csv에서 IS_PIT_LAP으로 이미 계산됨)
  2. 피트스탑을 마치고 나갈 때마다 stint_id를 1씩 증가시키고,
     다음 피트스탑 전까지 현재 달리고 있는 바퀴 수를 stint_lap으로 카운트
  3. stint_lap은 향후 모델에서 타이어 마모/연료 소모 대리 변수로 사용
"""
from __future__ import annotations
import pandas as pd


def reconstruct_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """
    laps: EVENT_ID, NUMBER(차량 번호), LAP_NUMBER, IS_PIT_LAP 컬럼을 포함해야 함.
    반환: STINT_ID, STINT_LAP 컬럼이 추가된 DataFrame (차량별로 정렬됨).
    """
    required = {"EVENT_ID", "NUMBER", "LAP_NUMBER", "IS_PIT_LAP"}
    missing = required - set(laps.columns)
    if missing:
        raise ValueError(f"reconstruct_stints: 필수 컬럼 누락 {missing}")

    laps = laps.sort_values(["EVENT_ID", "NUMBER", "LAP_NUMBER"]).reset_index(drop=True)

    out_chunks = []
    for (_event, _car), g in laps.groupby(["EVENT_ID", "NUMBER"], sort=False):
        g = g.copy()
        # 피트스탑 랩 "다음 랩"부터 새 stint 시작 -> shift한 IS_PIT_LAP의 누적합이 stint_id
        prev_pit = g["IS_PIT_LAP"].shift(1, fill_value=False)
        g["STINT_ID"] = prev_pit.cumsum().astype(int)
        # 각 stint 내에서 1부터 증가하는 stint_lap
        g["STINT_LAP"] = g.groupby("STINT_ID").cumcount() + 1
        out_chunks.append(g)

    result = pd.concat(out_chunks, ignore_index=True)
    return result


def compute_pit_loss(laps: pd.DataFrame, pit_loss_default_s: float) -> pd.DataFrame:
    """
    PIT_LOSS_S 우선순위:
      1. 실측값 (loaders.load_lap_csv에서 PIT_TIME을 파싱해 만든 PIT_LOSS_S_ACTUAL) — 가장 정확.
      2. 랩타임 기반 추정 (피트랩 랩타임 - 직전 3랩 평균) — PIT_TIME이 없는 경우의 대체.
      3. 상수 폴백 (pit_loss_default_s) — 위 둘 다 안 되는 경우.
    LAP_TIME은 'M:SS.mmm'/'H:MM:SS.mmm' 문자열이므로 time_utils로 파싱한다(pd.to_numeric 아님).
    """
    from time_utils import parse_time_str_to_seconds

    laps = laps.sort_values(["EVENT_ID", "NUMBER", "LAP_NUMBER"]).reset_index(drop=True).copy()
    laps["LAP_TIME_S"] = parse_time_str_to_seconds(laps.get("LAP_TIME", pd.Series(dtype=object)))

    chunks = []
    for _key, g in laps.groupby(["EVENT_ID", "NUMBER"], sort=False):
        g = g.copy()
        baseline = g["LAP_TIME_S"].rolling(3, min_periods=1).mean().shift(1)
        estimated = (g["LAP_TIME_S"] - baseline).where(g["IS_PIT_LAP"], other=pd.NA)
        chunks.append(g.assign(PIT_LOSS_S_ESTIMATED=estimated))

    laps = pd.concat(chunks, ignore_index=True)

    actual = laps.get("PIT_LOSS_S_ACTUAL")
    if actual is not None:
        laps["PIT_LOSS_S"] = actual.where(laps["IS_PIT_LAP"] & actual.notna(), other=laps["PIT_LOSS_S_ESTIMATED"])
    else:
        laps["PIT_LOSS_S"] = laps["PIT_LOSS_S_ESTIMATED"]
    laps["PIT_LOSS_S"] = laps["PIT_LOSS_S"].fillna(pit_loss_default_s)
    return laps
