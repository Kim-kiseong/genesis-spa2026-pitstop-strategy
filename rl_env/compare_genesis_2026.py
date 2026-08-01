"""
rl_env/compare_genesis_2026.py
=================================
Day5 - 김기성(RL Engineer) 담당.

학습(2021~2024)/평가(2025)에 전혀 쓰이지 않은 완전히 새로운 이벤트,
2026 SPA의 제네시스 마그마 레이싱 2대(#17, #19)를 대상으로:
  - 실제 팀의 피트인 전략(랩, 스틴트 길이, 트랙상태)
  - 규칙 기반 정책의 피트인 전략
  - PPO 정책의 피트인 전략
을 나란히 비교한다. 랩별 타임라인 차트로 저장.

실행:
    python -m rl_env.compare_genesis_2026
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import MaskablePPO

from evaluator.xgb_evaluator import load_evaluator
from rl_env.pitstop_env import PitstopEnv, load_race_laps
from rl_env.rule_based_strategy import DEFAULT_MODEL_PATH, run_rule_based_episode

PARQUET_PATH = Path(__file__).resolve().parent.parent / "preprocessing" / "laps_2026_spa.parquet"
PPO_MODEL_PATH = Path(__file__).resolve().parent / "outputs" / "ppo_pitstop.zip"
CHART_DIR = Path(__file__).resolve().parent / "outputs" / "charts"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN_GOOD = "#0ca30c"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
    }
)


def real_strategy(car: pd.DataFrame) -> dict:
    pit_laps = car.loc[car["IS_PIT_LAP"] == True, "LAP_NUMBER"].tolist()  # noqa: E712
    return {
        "pit_laps": pit_laps,
        "total_laps": int(car["LAP_NUMBER"].max()),
        "final_position": int(car["FINAL_CLASS_POSITION"].iloc[0]),
        "podium": bool(car["PODIUM"].iloc[0]),
    }


def rule_based_strategy_pits(race_laps: pd.DataFrame, evaluator) -> tuple[list[int], float]:
    env = PitstopEnv(race_laps=race_laps, evaluator=evaluator)
    result = run_rule_based_episode(env, max_steps=300)
    df = result.to_dataframe()
    return df.loc[df["ACTION"] == 1, "LAP"].tolist(), float(df["REWARD"].sum())


def ppo_strategy_pits(race_laps: pd.DataFrame, evaluator, ppo: MaskablePPO) -> tuple[list[int], float]:
    env = PitstopEnv(race_laps=race_laps, evaluator=evaluator)
    obs, _ = env.reset()
    pit_laps, cum_reward = [], 0.0
    for _ in range(300):
        mask = env.action_masks()
        action, _ = ppo.predict(obs, action_masks=mask, deterministic=True)
        action = int(action)
        if action == 1:
            pit_laps.append(env.current_lap)
        obs, reward, terminated, truncated, _ = env.step(action)
        cum_reward += reward
        if terminated or truncated:
            break
    return pit_laps, cum_reward


def compare_car(number: str, df: pd.DataFrame, evaluator, ppo: MaskablePPO) -> dict:
    car = df[df["NUMBER"] == number].sort_values("LAP_NUMBER")
    real = real_strategy(car)

    race_laps = load_race_laps(PARQUET_PATH, event_id="2026_SPA", number=number)
    rule_pits, rule_reward = rule_based_strategy_pits(race_laps, evaluator)
    ppo_pits, ppo_reward = ppo_strategy_pits(race_laps, evaluator, ppo)

    return {
        "number": number,
        "real": real,
        "rule_pits": rule_pits,
        "rule_reward": rule_reward,
        "ppo_pits": ppo_pits,
        "ppo_reward": ppo_reward,
    }


def plot_timeline(results: list[dict], out_path: Path):
    fig, axes = plt.subplots(len(results), 1, figsize=(9, 2.4 * len(results) + 1), sharex=False)
    if len(results) == 1:
        axes = [axes]

    rows = [("실제 팀", "real"), ("규칙 기반", "rule"), ("PPO", "ppo")]
    colors = {"real": INK, "rule": BLUE, "ppo": ORANGE}

    for ax, res in zip(axes, results):
        total_laps = res["real"]["total_laps"]
        y_positions = {"real": 2, "rule": 1, "ppo": 0}
        pit_lists = {
            "real": res["real"]["pit_laps"],
            "rule": res["rule_pits"],
            "ppo": res["ppo_pits"],
        }

        ax.hlines(
            [2, 1, 0], 0, total_laps, color=GRID, linewidth=6, zorder=1, capstyle="round"
        )
        for key, y in y_positions.items():
            laps = pit_lists[key]
            ax.scatter(laps, [y] * len(laps), s=90, color=colors[key], zorder=3, edgecolor=SURFACE, linewidth=1)

        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["실제 팀", "규칙 기반", "PPO"])
        ax.set_xlim(0, total_laps + 2)
        ax.set_xlabel("랩 번호" if res is results[-1] else "")
        podium_txt = "포디움" if res["real"]["podium"] else f"{res['real']['final_position']}위"
        ax.set_title(
            f"제네시스 #{res['number']} — 실제 {len(pit_lists['real'])}회 / "
            f"규칙 {len(pit_lists['rule'])}회 / PPO {len(pit_lists['ppo'])}회  (실제 최종: {podium_txt})",
            fontsize=11.5, fontweight="bold", loc="left",
        )
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)

    fig.suptitle("2026 SPA 제네시스 마그마 레이싱 — 피트인 타임라인 비교", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_parquet(PARQUET_PATH)
    evaluator = load_evaluator(model_path=DEFAULT_MODEL_PATH)
    ppo = MaskablePPO.load(PPO_MODEL_PATH)

    results = [compare_car(num, df, evaluator, ppo) for num in ["17", "19"]]

    for res in results:
        r = res["real"]
        print(f"\n===== 제네시스 #{res['number']} =====")
        print(f"실제 결과: {r['final_position']}위 (포디움: {r['podium']})")
        print(f"실제 피트({len(r['pit_laps'])}회): {r['pit_laps']}")
        print(f"규칙 기반 피트({len(res['rule_pits'])}회): {res['rule_pits']} | 누적보상 {res['rule_reward']:+.4f}")
        print(f"PPO 피트({len(res['ppo_pits'])}회): {res['ppo_pits']} | 누적보상 {res['ppo_reward']:+.4f}")

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHART_DIR / "genesis_2026_pit_timeline.png"
    plot_timeline(results, out_path)
    print(f"\n차트 저장: {out_path}")
