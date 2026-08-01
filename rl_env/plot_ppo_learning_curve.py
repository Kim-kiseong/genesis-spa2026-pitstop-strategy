"""
rl_env/plot_ppo_learning_curve.py
====================================
Day5 - 김기성(RL Engineer) 담당.

train_ppo.py 실행 로그(콘솔에 찍힌 SB3 rollout 통계)에서 ep_rew_mean(에피소드
평균 보상)과 total_timesteps을 뽑아 학습 곡선으로 그린다. "PPO 에이전트
학습 가능성(Learning Curve) 정리" 항목용.

로그 파일이 없으면(재학습 안 했으면) train_ppo.train()을 다시 돌려서
학습 곡선용 로그를 새로 뽑을 수도 있지만, 보통은 이미 있는 학습 실행의
콘솔 출력을 텍스트 파일로 저장해서 이 스크립트로 파싱하는 쪽이 빠르다.

실행:
    python -m rl_env.plot_ppo_learning_curve <로그파일경로>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHART_DIR = Path(__file__).resolve().parent / "outputs" / "charts"

BLUE = "#2a78d6"
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

# SB3 콘솔 출력 한 iteration 블록에서 ep_rew_mean과 total_timesteps을 같이 뽑는다.
PATTERN = re.compile(
    r"ep_rew_mean\s*\|\s*(-?[\d.]+)\s*\|.*?total_timesteps\s*\|\s*(\d+)",
    re.S,
)


def parse_log(log_path: Path) -> list[tuple[int, float]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    rows = [(int(ts), float(rew)) for rew, ts in PATTERN.findall(text)]
    if not rows:
        raise ValueError(f"{log_path}에서 ep_rew_mean/total_timesteps을 못 찾았습니다.")
    return rows


def plot_curve(rows: list[tuple[int, float]], out_path: Path):
    steps = [r[0] for r in rows]
    rewards = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(steps, rewards, color=BLUE, linewidth=2, marker="o", markersize=6, zorder=3)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1, linestyle="--", zorder=2)

    ax.annotate(
        f"{rewards[0]:+.2f}", (steps[0], rewards[0]),
        textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10,
    )
    ax.annotate(
        f"{rewards[-1]:+.2f}", (steps[-1], rewards[-1]),
        textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10, fontweight="bold",
    )

    ax.set_xlabel("학습 스텝 (total_timesteps)")
    ax.set_title("PPO 학습 곡선 — 에피소드 평균 보상(ep_rew_mean)", fontsize=13.5, fontweight="bold", pad=30)
    ax.text(
        0.0, 1.05, "ep_rew_mean (10에피소드 이동평균)", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=10.5, color=INK_SECONDARY,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python -m rl_env.plot_ppo_learning_curve <로그파일경로>")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    rows = parse_log(log_path)

    print(f"{len(rows)}개 iteration 파싱됨:")
    for ts, rew in rows:
        print(f"  step={ts:>6}  ep_rew_mean={rew:+.4f}")

    CHART_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHART_DIR / "ppo_learning_curve.png"
    plot_curve(rows, out_path)
    print(f"\n차트 저장: {out_path}")
