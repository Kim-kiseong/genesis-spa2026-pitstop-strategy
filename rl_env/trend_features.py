"""
rl_env/trend_features.py
==========================
Day4 - 김기성(RL Engineer) 담당.

최근 N랩(기본 5랩, evaluator.schema.TREND_WINDOW_LAPS) 슬라이딩 윈도우로
단일 시계열(순위 또는 격차)의 변화량과 추세를 계산한다.

- change: 윈도우 마지막 값 - 첫 값 (단순 차분)
- slope:  윈도우 값들에 선형회귀(최소제곱)를 적합했을 때의 기울기

PitstopEnv가 실제 레이스 랩을 한 랩씩 재생하며 매 스텝 push()로 값을 넣고,
change_and_slope()를 호출해 pos_change_5lap 등 4개 피처를 얻는 용도로 쓴다.
레이스 초반이라 윈도우가 아직 안 찼으면(랩 수 < window) (0.0, 0.0)을 반환한다.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class SlidingTrendWindow:
    """단일 시계열에 대한 고정 크기 슬라이딩 윈도우 + 선형회귀 기울기 계산기."""

    def __init__(self, window: int = 5):
        self.window = window
        self._buf: deque[float] = deque(maxlen=window)

    def push(self, value: float) -> None:
        self._buf.append(float(value))

    def reset(self) -> None:
        self._buf.clear()

    def change_and_slope(self) -> tuple[float, float]:
        if len(self._buf) < self.window:
            return 0.0, 0.0
        values = np.asarray(self._buf, dtype=np.float64)
        change = float(values[-1] - values[0])
        x = np.arange(self.window, dtype=np.float64)
        slope = float(np.polyfit(x, values, 1)[0])
        return change, slope
