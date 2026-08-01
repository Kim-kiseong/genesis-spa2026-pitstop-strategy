"""
rl_env/sweep_baseline.py
===========================
Day4 - 김기성(RL Engineer) 담당.

PODIUM_PROB_THRESHOLD 같은 규칙 임계값을 차량 1대 감으로 정하지 않기 위한 스윕 스크립트.
val 세트의 모든 차량 x 여러 임계값 후보에 대해 규칙 기반 에피소드를 반복 실행하고,
후보값별로 결과를 요약해서 비교 근거를 만든다.

실행:
    python -m rl_env.sweep_baseline
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluator.schema import MAX_STINT_LAPS
from evaluator.xgb_evaluator import load_evaluator
from rl_env.pitstop_env import PitstopEnv, load_race_laps, DEFAULT_VAL_PARQUET
from rl_env.rule_based_strategy import (
    DEFAULT_MODEL_PATH,
    PODIUM_PROB_THRESHOLD,
    TIRE_WEAR_THRESHOLD,
    run_rule_based_episode,
)

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# 기획서 기본값(0.7)을 포함해 위아래로 후보를 잡는다.
THRESHOLD_CANDIDATES: list[float] = [0.5, 0.6, PODIUM_PROB_THRESHOLD, 0.8]


def list_cars(parquet_path: str | Path = DEFAULT_VAL_PARQUET) -> pd.DataFrame:
    """val 세트에 있는 (EVENT_ID, NUMBER) 차량-이벤트 조합과 실제 결과를 뽑는다."""
    df = pd.read_parquet(
        parquet_path, columns=["EVENT_ID", "NUMBER", "FINAL_CLASS_POSITION", "PODIUM"]
    )
    return df.drop_duplicates(["EVENT_ID", "NUMBER"]).reset_index(drop=True)


def sweep(
    parquet_path: str | Path = DEFAULT_VAL_PARQUET,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold_candidates: list[float] = THRESHOLD_CANDIDATES,
    tire_wear_threshold: float = TIRE_WEAR_THRESHOLD,
) -> pd.DataFrame:
    """차량 x 임계값 후보 조합마다 한 번씩 에피소드를 돌려서 행 하나로 요약한다."""
    evaluator = load_evaluator(model_path=model_path)
    cars = list_cars(parquet_path)

    rows = []
    for threshold in threshold_candidates:
        for _, car in cars.iterrows():
            race_laps = load_race_laps(parquet_path, event_id=car["EVENT_ID"], number=car["NUMBER"])
            env = PitstopEnv(race_laps=race_laps, evaluator=evaluator)
            result = run_rule_based_episode(
                env,
                podium_prob_threshold=threshold,
                tire_wear_threshold=tire_wear_threshold,
            )
            df = result.to_dataframe()
            pit_rows = df[df["ACTION"] == 1]

            rows.append(
                {
                    "PODIUM_PROB_THRESHOLD": threshold,
                    "EVENT_ID": car["EVENT_ID"],
                    "NUMBER": car["NUMBER"],
                    "REAL_FINAL_POSITION": car["FINAL_CLASS_POSITION"],
                    "REAL_PODIUM": bool(car["PODIUM"]),
                    "SIM_LAPS": len(df),
                    "SIM_PIT_COUNT": len(pit_rows),
                    "SIM_EARLY_PIT_COUNT": int((pit_rows["STINT_LAP"] < MAX_STINT_LAPS).sum()),
                    "SIM_MEAN_PODIUM_PROB": df["PODIUM_PROB"].mean(),
                    "SIM_CUM_REWARD": df["REWARD"].sum(),
                }
            )

    return pd.DataFrame(rows)


def summarize(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """임계값 후보별로 여러 차량 결과를 집계 — 이 표가 임계값을 고르는 근거가 된다."""
    return (
        sweep_df.groupby("PODIUM_PROB_THRESHOLD")
        .agg(
            차량수=("NUMBER", "count"),
            평균피트횟수=("SIM_PIT_COUNT", "mean"),
            평균조기피트횟수=("SIM_EARLY_PIT_COUNT", "mean"),
            조기피트발생차량비율=("SIM_EARLY_PIT_COUNT", lambda s: (s > 0).mean()),
            평균누적보상=("SIM_CUM_REWARD", "mean"),
        )
        .reset_index()
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    print(f"모델: {DEFAULT_MODEL_PATH}")
    print(f"임계값 후보: {THRESHOLD_CANDIDATES}")

    sweep_df = sweep()
    sweep_df.to_csv(OUT_DIR / "threshold_sweep_raw.csv", index=False)

    summary_df = summarize(sweep_df)
    summary_df.to_csv(OUT_DIR / "threshold_sweep_summary.csv", index=False)

    n_cars = sweep_df["NUMBER"].nunique()
    print(f"\n스윕 완료: 차량 {n_cars}대 x 임계값 후보 {len(THRESHOLD_CANDIDATES)}개")
    print(summary_df.to_string(index=False))
    print(f"\n원본(차량별 전체 결과): {OUT_DIR / 'threshold_sweep_raw.csv'}")
    print(f"요약(임계값별 집계):     {OUT_DIR / 'threshold_sweep_summary.csv'}")
