"""
Day 1~3 오케스트레이터 — 팀원 A(Data Engineer) 최종 산출물.

원시 CSV -> 스틴트 복원(Day1) -> 날씨 결합(Day2) -> 경쟁상황/트랙상태 -> 이상치 제거,
train/val 스플릿(Day3) -> processed/laps_train.parquet, processed/laps_val.parquet

B, C에게 넘길 인터페이스는 이 두 파일과 README.md의 컬럼 설명이 전부입니다.

실행:
    python build_dataset.py
"""
from __future__ import annotations
import pandas as pd

from config import (
    RAW_DIR, PROCESSED_DIR, TRAIN_EVENTS, VAL_EVENTS, CLASS_FILTER,
    PIT_LOSS_DEFAULT_S, PODIUM_RANK_THRESHOLD, MAX_STINT_LAPS,
)
from loaders import load_event
from stint import reconstruct_stints, compute_pit_loss
from weather import merge_weather
from competition import reconstruct_competition_state, compute_lap_progress_ratio
from track_status import apply_flag_at_fl
from outliers import drop_track_status_outliers, flag_pit_loss_outliers, drop_degenerate_podium_events


def process_event(event: dict) -> pd.DataFrame | None:
    event_id = f"{event['year']}_{event['circuit']}"
    try:
        laps, weather, classification = load_event(event, RAW_DIR, class_filter=CLASS_FILTER)
    except FileNotFoundError as e:
        print(f"  [건너뜀] {event_id}: {e}")
        return None

    # Day 1: 스틴트 복원
    laps = reconstruct_stints(laps)
    laps = compute_pit_loss(laps, PIT_LOSS_DEFAULT_S)

    # 교수님 피드백 반영: 스틴트 진행률(0~1 정규화) — 서킷/경기시간이 달라도 비교 가능하게.
    laps["STINT_PROGRESS_RATIO"] = (laps["STINT_LAP"] / MAX_STINT_LAPS).clip(upper=1.0)

    # Day 2: 날씨 결합
    laps = merge_weather(laps, weather, circuit=event["circuit"], event_id=event_id)

    # 경쟁 상황 + 트랙 상태 (RF/FF 실데이터 검증 완료)
    laps = reconstruct_competition_state(laps)
    laps = compute_lap_progress_ratio(laps)
    laps = apply_flag_at_fl(laps)

    # 포디움 라벨 (Classification 파일 없으면 자체 계산 마지막 랩 순위로 근사)
    if classification is not None and "FINAL_CLASS_POSITION" in classification.columns:
        final_pos = classification[["NUMBER", "FINAL_CLASS_POSITION"]].drop_duplicates("NUMBER")
        laps = laps.merge(final_pos, on="NUMBER", how="left")
    else:
        last_lap = laps.sort_values("LAP_NUMBER").groupby("NUMBER").tail(1)[["NUMBER", "CLASS_POSITION"]]
        last_lap = last_lap.rename(columns={"CLASS_POSITION": "FINAL_CLASS_POSITION"})
        laps = laps.merge(last_lap, on="NUMBER", how="left")
    laps["PODIUM"] = laps["FINAL_CLASS_POSITION"] <= PODIUM_RANK_THRESHOLD

    return laps


def build_split(events: list[dict], split_name: str) -> pd.DataFrame:
    print(f"[{split_name}] {len(events)}개 이벤트 처리 중...")
    all_laps, all_dropped = [], []
    for event in events:
        laps = process_event(event)
        if laps is None:
            continue
        laps, dropped = drop_track_status_outliers(laps)  # Day 3: 레드플래그 랩 드롭
        laps = flag_pit_loss_outliers(laps)                # Day 3: 비정상 피트로스 플래그만
        all_laps.append(laps)
        all_dropped.append(dropped)

    if not all_laps:
        raise RuntimeError(f"[{split_name}] 처리된 이벤트가 없습니다. raw_data/를 확인하세요.")

    result = pd.concat(all_laps, ignore_index=True)

    # [보류 — 팀 내부 회의 후 결정 예정] 참가 차량이 너무 적어 PODIUM 라벨이 왜곡된 이벤트
    # (실측: 2021·2022 SPA 전원 포디움) 제외 여부. 함수는 outliers.py에 준비돼 있으니
    # 팀 결정 나면 아래 두 줄 주석만 풀면 바로 적용됨.
    # result, degenerate = drop_degenerate_podium_events(result)
    # if len(degenerate):
    #     degenerate.to_parquet(PROCESSED_DIR / f"dropped_degenerate_podium_{split_name}.parquet", index=False)
    #     print(f"  라벨 왜곡으로 드롭된 랩: {len(degenerate)}행 -> dropped_degenerate_podium_{split_name}.parquet")

    dropped_total = pd.concat(all_dropped, ignore_index=True) if all_dropped else pd.DataFrame()
    if len(dropped_total):
        dropped_total.to_parquet(PROCESSED_DIR / f"dropped_redflag_{split_name}.parquet", index=False)
        print(f"  레드플래그로 드롭된 랩: {len(dropped_total)}행 -> dropped_redflag_{split_name}.parquet (케이스 스터디용 보관)")

    print(f"  {split_name} 완료: {len(result)}행")
    return result


def main():
    train = build_split(TRAIN_EVENTS, "train")
    val = build_split(VAL_EVENTS, "val")

    train.to_parquet(PROCESSED_DIR / "laps_train.parquet", index=False)
    val.to_parquet(PROCESSED_DIR / "laps_val.parquet", index=False)

    print(f"\n최종 산출물:")
    print(f"  {PROCESSED_DIR / 'laps_train.parquet'} ({len(train)} rows)")
    print(f"  {PROCESSED_DIR / 'laps_val.parquet'} ({len(val)} rows)")


if __name__ == "__main__":
    main()
