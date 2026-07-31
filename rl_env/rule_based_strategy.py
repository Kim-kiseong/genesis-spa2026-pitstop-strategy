"""
rl_env/rule_based_strategy.py
================================
Day 3 - 김기성(RL Engineer) 담당.

교수님 피드백 4번 반영: PPO 학습에 들어가기 전에 빠르고 확실한
if-else 규칙 기반 시뮬레이터로 기준점(Baseline)을 먼저 잡는다.

evaluator/xgb_evaluator.py::load_evaluator()로 B(주윤서)가 전달한 실제
XGBoost 모델(evaluator/xgb_podium_model.joblib)을 PitstopEnv에 연동하고,
아래 규칙으로 랩마다 피트인 여부를 결정해 한 에피소드를 시뮬레이션한다.

규칙 (기획서 예시 그대로):
    포디움 확률 >= PODIUM_PROB_THRESHOLD 이고
    타이어 마모율(STINT_LAP / MAX_STINT_LAPS) >= TIRE_WEAR_THRESHOLD 일 때만 피트인.
    포디움 경쟁권이 아니면 피트로스를 감수할 이유가 없으므로,
    MAX_STINT_LAPS에 의한 강제 피트까지 그대로 밀어붙인다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from evaluator.schema import MAX_STINT_LAPS, MIN_STINT_LAPS
from evaluator.xgb_evaluator import load_evaluator
from rl_env.pitstop_env import PitstopEnv

PODIUM_PROB_THRESHOLD = 0.7
TIRE_WEAR_THRESHOLD = 0.8


def should_pit(podium_prob: float, stint_lap: int) -> int:
    """규칙 기반 피트인 결정. 반환값 1=피트인, 0=계속 주행."""
    if stint_lap < MIN_STINT_LAPS:
        return 0
    if stint_lap >= MAX_STINT_LAPS:
        return 1

    tire_wear_ratio = stint_lap / MAX_STINT_LAPS
    if podium_prob >= PODIUM_PROB_THRESHOLD and tire_wear_ratio >= TIRE_WEAR_THRESHOLD:
        return 1
    return 0


@dataclass
class RuleBasedResult:
    laps: list[int] = field(default_factory=list)
    stint_laps: list[int] = field(default_factory=list)
    podium_probs: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "LAP": self.laps,
                "STINT_LAP": self.stint_laps,
                "PODIUM_PROB": self.podium_probs,
                "ACTION": self.actions,
                "REWARD": self.rewards,
            }
        )


def run_rule_based_episode(env: PitstopEnv, max_steps: int = 200) -> RuleBasedResult:
    """규칙 기반 정책으로 한 에피소드를 끝까지 시뮬레이션한다."""
    obs, _ = env.reset()
    result = RuleBasedResult()

    for _ in range(max_steps):
        action = should_pit(env.current_podium_prob, env.stint_lap)

        # 규칙과 액션 마스킹이 어긋나는 예외 상황(임계값 경계 등) 방어
        mask = env.action_masks()
        if not mask[action]:
            action = 1 - action

        result.laps.append(env.current_lap)
        result.stint_laps.append(env.stint_lap)
        result.podium_probs.append(env.current_podium_prob)
        result.actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        result.rewards.append(reward)

        if terminated or truncated:
            break

    return result


if __name__ == "__main__":
    evaluator = load_evaluator()
    env = PitstopEnv(evaluator=evaluator)

    result = run_rule_based_episode(env)
    df = result.to_dataframe()

    print(f"\n규칙 기반 시뮬레이션 완료: {len(df)}랩")
    print(f"피트인 횟수: {int(df['ACTION'].sum())}회")
    print(f"누적 보상(포디움 확률 변화 합): {df['REWARD'].sum():+.4f}")
    print(df.head(10).to_string(index=False))

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "rule_based_baseline.csv"
    df.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")
