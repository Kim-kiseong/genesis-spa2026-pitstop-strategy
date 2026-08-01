"""
rl_env/make_report_charts.py
===============================
Day4 - 김기성(RL Engineer) 담당.

보고서용 정적 차트(PNG) 생성. rl_env/outputs/의 CSV 결과를 읽어서
rl_env/outputs/charts/에 4장을 저장한다:
  1. pit_penalty_effect.png   - 리워드 해킹 전/후 평균 피트횟수
  2. reward_by_podium.png     - 포디움 성공/실패 차량별 평균 누적보상(규칙 vs PPO)
  3. threshold_sweep.png      - PODIUM_PROB_THRESHOLD별 평균 누적보상
  4. ppo_vs_rule_paired.png   - 차량별 PPO 대 규칙 기반 페어 비교(대각선 위 = PPO 승)

실행:
    python -m rl_env.make_report_charts
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "outputs"
CHART_DIR = OUT_DIR / "charts"

# 팔레트 (dataviz 스킬 참고 팔레트)
BLUE = "#2a78d6"      # 카테고리 슬롯1 - 규칙 기반
ORANGE = "#eb6834"    # 카테고리 슬롯2 - PPO
GREEN_GOOD = "#0ca30c"
RED_CRITICAL = "#d03b3b"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",  # 한글 지원 (Windows 기본)
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
    }
)


def _style_axes(ax, y_grid=True):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if y_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _y_unit_label(ax, text: str):
    """세로로 회전된 한글 y축 라벨은 렌더링이 지저분해서, 축 위쪽에 가로 텍스트로 대신 붙인다."""
    ax.text(
        0.0, 1.03, text, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=10.5, color=INK_SECONDARY,
    )


def chart_pit_penalty_effect():
    """리워드 해킹 전/후 평균 피트횟수 (규칙 기반 기준선 포함)."""
    ppo_before = 44.42
    rule_baseline = 8.28
    ppo_after = 8.31

    labels = ["PPO\n(페널티 적용 전)", "규칙 기반\n(기준값)", "PPO\n(페널티 적용 후)"]
    values = [ppo_before, rule_baseline, ppo_after]
    colors = [RED_CRITICAL, BLUE, GREEN_GOOD]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1,
            f"{v:.2f}회",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )

    ax.set_title("피트인 페널티 도입 전/후 — 리워드 해킹 교정", fontsize=14, fontweight="bold", pad=22)
    ax.set_ylim(0, 52)
    _style_axes(ax)
    _y_unit_label(ax, "평균 피트횟수 (val 36개 차량)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "pit_penalty_effect.png", dpi=180)
    plt.close(fig)


def chart_reward_by_podium():
    """포디움 성공/실패 차량별 평균 누적보상 - 규칙 기반 vs PPO."""
    groups = ["실제 포디움 실패 차량\n(30대)", "실제 포디움 성공 차량\n(6대)"]
    rule_vals = [-0.490, -0.243]
    ppo_vals = [-0.519, 0.027]

    x = range(len(groups))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 5.5))
    b1 = ax.bar([i - width / 2 for i in x], rule_vals, width, label="규칙 기반", color=BLUE, zorder=3)
    b2 = ax.bar([i + width / 2 for i in x], ppo_vals, width, label="PPO", color=ORANGE, zorder=3)

    # 라벨은 항상 막대 끝(0에 가까운 쪽) 바깥에 붙여서, 음수 막대라도 x축 눈금 라벨과 안 겹치게 한다.
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + (0.012 if v >= 0 else -0.012),
                f"{v:+.3f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=10.5,
                color=INK,
            )

    ax.axhline(0, color=INK_MUTED, linewidth=1, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(-0.62, 0.14)
    ax.set_title("실제 결과별 평균 누적보상 — PPO가 포디움 차량에서 우위", fontsize=13.5, fontweight="bold", pad=22)
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    _y_unit_label(ax, "평균 누적보상")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "reward_by_podium.png", dpi=180)
    plt.close(fig)


def chart_threshold_sweep():
    """PODIUM_PROB_THRESHOLD 후보별 평균 누적보상 (규칙 기반, val 36개 차량)."""
    df = pd.read_csv(OUT_DIR / "threshold_sweep_summary.csv")
    df = df.sort_values("PODIUM_PROB_THRESHOLD")
    x = df["PODIUM_PROB_THRESHOLD"]
    y = df["평균누적보상"]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=8, zorder=3)
    for xi, yi in zip(x, y):
        label_default = "  (기획서 기본값)" if abs(xi - 0.7) < 1e-9 else ""
        ax.annotate(
            f"{yi:.4f}{label_default}",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color=INK,
        )

    ax.set_xlabel("PODIUM_PROB_THRESHOLD 후보")
    ax.set_title("임계값 후보별 평균 누적보상 — 낮출수록 오히려 악화", fontsize=13.5, fontweight="bold", pad=22)
    ax.set_xticks(x)
    ax.set_ylim(y.min() - 0.006, y.max() + 0.006)
    _style_axes(ax)
    _y_unit_label(ax, "평균 누적보상 (val 36개 차량)")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "threshold_sweep.png", dpi=180)
    plt.close(fig)


def chart_ppo_vs_rule_paired():
    """차량별 PPO vs 규칙 기반 페어 비교 산점도 (대각선 위 = PPO 승)."""
    rule = pd.read_csv(OUT_DIR / "threshold_sweep_raw.csv", dtype={"NUMBER": str})
    rule07 = rule[rule["PODIUM_PROB_THRESHOLD"] == 0.7][
        ["EVENT_ID", "NUMBER", "SIM_CUM_REWARD", "REAL_PODIUM"]
    ].rename(columns={"SIM_CUM_REWARD": "RULE_REWARD"})
    ppo = pd.read_csv(OUT_DIR / "ppo_val_eval.csv", dtype={"NUMBER": str})[
        ["EVENT_ID", "NUMBER", "SIM_CUM_REWARD"]
    ].rename(columns={"SIM_CUM_REWARD": "PPO_REWARD"})
    merged = rule07.merge(ppo, on=["EVENT_ID", "NUMBER"], validate="one_to_one")

    non_podium = merged[~merged["REAL_PODIUM"]]
    podium = merged[merged["REAL_PODIUM"]]

    lo = min(merged["RULE_REWARD"].min(), merged["PPO_REWARD"].min()) - 0.08
    hi = max(merged["RULE_REWARD"].max(), merged["PPO_REWARD"].max()) + 0.08

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1.5, linestyle="--", zorder=2)
    ax.scatter(
        non_podium["RULE_REWARD"], non_podium["PPO_REWARD"],
        s=55, color=BLUE, alpha=0.75, edgecolor=SURFACE, linewidth=0.6,
        label="실제 포디움 실패 (30대)", zorder=3,
    )
    ax.scatter(
        podium["RULE_REWARD"], podium["PPO_REWARD"],
        s=95, color=GREEN_GOOD, edgecolor=SURFACE, linewidth=0.8,
        label="실제 포디움 성공 (6대)", zorder=4,
    )

    ax.text(
        0.03, 0.97, "PPO 승 (대각선 위)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, color=INK_SECONDARY, style="italic",
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("규칙 기반 누적보상")
    ax.set_title("차량별 1:1 비교 — 포디움 성공 차량 6/6 전부 PPO 승", fontsize=13, fontweight="bold", pad=32)
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax, y_grid=False)
    _y_unit_label(ax, "PPO 누적보상")
    fig.subplots_adjust(top=0.88)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ppo_vs_rule_paired.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    chart_pit_penalty_effect()
    chart_reward_by_podium()
    chart_threshold_sweep()
    chart_ppo_vs_rule_paired()

    print(f"차트 4장 저장 완료: {CHART_DIR}")
    for f in sorted(CHART_DIR.glob("*.png")):
        print(f"  - {f.name}")
