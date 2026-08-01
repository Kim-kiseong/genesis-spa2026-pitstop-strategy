"""
rl_env/train_ppo.py
======================
Day4 - 김기성(RL Engineer) 담당.

MaskablePPO PoC 학습. rl_env/rule_based_strategy.py의 규칙 기반 정책(기준점,
Baseline)보다 나은 피트인 전략을 PPO가 스스로 배울 수 있는지 확인하는 첫 시도.

학습은 laps_train.parquet의 58개 차량-이벤트(2021~2024 SPA/IMOLA)를 매 에피소드
무작위로 겪으며 진행한다. laps_val.parquet(학습에 안 쓰인 2025 SPA/IMOLA)은
규칙 기반과의 비교 평가에만 쓴다 — B의 XGBoost train/val 분리와 같은 이유로,
PPO가 평가 레이스를 외워서 성능이 부풀려지는 걸 막기 위함.

실행:
    python -m rl_env.train_ppo
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from evaluator.xgb_evaluator import load_evaluator
from rl_env.pitstop_env import PitstopEnv, load_race_laps, load_race_pool, DEFAULT_TRAIN_PARQUET
from rl_env.rule_based_strategy import DEFAULT_MODEL_PATH
from rl_env.sweep_baseline import list_cars

OUT_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_OUT_PATH = OUT_DIR / "ppo_pitstop.zip"

TOTAL_TIMESTEPS = 20_000  # PoC 규모 — 수렴 확인용이 아니라 "학습이 되는지"부터 확인


def mask_fn(env: PitstopEnv):
    return env.action_masks()


def make_train_env(evaluator) -> ActionMasker:
    race_pool = load_race_pool(DEFAULT_TRAIN_PARQUET)
    env = PitstopEnv(race_laps=race_pool, evaluator=evaluator)
    return ActionMasker(env, mask_fn)


def train(total_timesteps: int = TOTAL_TIMESTEPS) -> MaskablePPO:
    evaluator = load_evaluator(model_path=DEFAULT_MODEL_PATH)
    env = make_train_env(evaluator)

    model = MaskablePPO("MlpPolicy", env, verbose=1, seed=42)
    model.learn(total_timesteps=total_timesteps)

    OUT_DIR.mkdir(exist_ok=True)
    model.save(MODEL_OUT_PATH)
    print(f"\nPPO 모델 저장: {MODEL_OUT_PATH}")
    return model


def evaluate_on_val(model: MaskablePPO, evaluator, max_steps: int = 300) -> pd.DataFrame:
    """val 세트 36개 차량에 대해 학습된 정책을 결정론적으로 굴려서 규칙 기반과 비교 가능한 지표를 뽑는다."""
    cars = list_cars()
    rows = []

    for _, car in cars.iterrows():
        race_laps = load_race_laps(event_id=car["EVENT_ID"], number=car["NUMBER"])
        env = PitstopEnv(race_laps=race_laps, evaluator=evaluator)
        obs, _ = env.reset()

        cum_reward, pit_count, laps = 0.0, 0, 0
        for _ in range(max_steps):
            action_masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            cum_reward += reward
            pit_count += int(action)
            laps += 1
            if terminated or truncated:
                break

        rows.append(
            {
                "EVENT_ID": car["EVENT_ID"],
                "NUMBER": car["NUMBER"],
                "REAL_FINAL_POSITION": car["FINAL_CLASS_POSITION"],
                "REAL_PODIUM": bool(car["PODIUM"]),
                "SIM_LAPS": laps,
                "SIM_PIT_COUNT": pit_count,
                "SIM_CUM_REWARD": cum_reward,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    model = train()

    evaluator = load_evaluator(model_path=DEFAULT_MODEL_PATH)
    eval_df = evaluate_on_val(model, evaluator)

    OUT_DIR.mkdir(exist_ok=True)
    eval_df.to_csv(OUT_DIR / "ppo_val_eval.csv", index=False)

    print(f"\nPPO val 평가 완료: {len(eval_df)}대 차량")
    print(f"평균 피트횟수: {eval_df['SIM_PIT_COUNT'].mean():.2f}")
    print(f"평균 누적보상: {eval_df['SIM_CUM_REWARD'].mean():+.4f}")
    print(eval_df.groupby("REAL_PODIUM")[["SIM_PIT_COUNT", "SIM_CUM_REWARD"]].mean().round(3))
    print(f"\n결과 저장: {OUT_DIR / 'ppo_val_eval.csv'}")
