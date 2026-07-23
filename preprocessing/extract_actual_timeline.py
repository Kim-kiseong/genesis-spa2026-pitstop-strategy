"""
Day 4. B, C에게 넘길 "실제 피트 타임라인" 추출 스크립트.

C가 "실제 팀은 X랩에 피트했는데 AI는 Y랩에 피트해서 포디움 확률이 Z% 올랐다"는
비교를 만들 때 쓰는 실제 정답 데이터. laps_val.parquet에서 특정 차량 하나를 뽑아
랩별 피트 여부/스틴트/포디움 라벨을 정리해 CSV로 저장한다.

실행 예:
    python extract_actual_timeline.py --event 2025_SPA --number 17
"""
from __future__ import annotations
import argparse
import pandas as pd

from config import PROCESSED_DIR


def extract(event_id: str, number: str) -> pd.DataFrame:
    val = pd.read_parquet(PROCESSED_DIR / "laps_val.parquet")
    car = val[(val["EVENT_ID"] == event_id) & (val["NUMBER"].astype(str) == str(number))]
    car = car.sort_values("LAP_NUMBER")
    if car.empty:
        raise ValueError(f"{event_id}의 차량 번호 {number}를 찾을 수 없습니다. (laps_val.parquet에 없음)")

    cols = [c for c in [
        "LAP_NUMBER", "STINT_ID", "STINT_LAP", "IS_PIT_LAP", "PIT_LOSS_S",
        "TRACK_STATUS", "WEATHER_CATEGORY", "TRACK_TEMP", "CLASS_POSITION",
        "GAP_TO_LEADER_SEC", "GAP_TO_AHEAD_SEC", "LAP_PROGRESS_RATIO",
        "FINAL_CLASS_POSITION", "PODIUM",
    ] if c in car.columns]
    return car[cols].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, help="예: 2025_SPA")
    parser.add_argument("--number", required=True, help="차량 번호, 예: 17")
    args = parser.parse_args()

    timeline = extract(args.event, args.number)
    out_path = PROCESSED_DIR / f"actual_timeline_{args.event}_{args.number}.csv"
    timeline.to_csv(out_path, index=False)

    pit_laps = timeline.loc[timeline["IS_PIT_LAP"], "LAP_NUMBER"].tolist()
    print(f"{args.event} #{args.number}: 총 {len(timeline)}랩, 실제 피트 랩 {pit_laps}")
    print(f"저장: {out_path}")
