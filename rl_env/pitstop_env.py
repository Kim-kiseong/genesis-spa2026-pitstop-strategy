from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# 팀원 B가 작성한 evaluator 모듈 임포트
from evaluator.schema import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_TREND,
    MIN_STINT_LAPS,
    MAX_STINT_LAPS,
    TREND_WINDOW_LAPS,
)
from evaluator.dummy_evaluator import DummyPodiumEvaluator
from rl_env.trend_features import SlidingTrendWindow

DEFAULT_VAL_PARQUET = (
    Path(__file__).resolve().parent.parent / "preprocessing" / "processed" / "laps_val.parquet"
)


def load_race_laps(
    parquet_path: str | Path = DEFAULT_VAL_PARQUET,
    event_id: str | None = None,
    number: int | str | None = None,
) -> pd.DataFrame:
    """
    laps_train.parquet / laps_val.parquet에서 특정 차량 한 대의 레이스 랩 시퀀스를 읽어온다.
    event_id, number를 안 주면 파일에서 처음 발견되는 (EVENT_ID, NUMBER) 조합을 그대로 쓴다
    (Day4 "실제 레이싱 팀 vs 규칙 기반" 비교용으로 특정 차량을 명시적으로 골라 넘기는 게 일반적).
    PitstopEnv가 요구하는 LAP_NUMBER + FEATURE_COLUMNS 컬럼만 남겨서 반환한다.
    """
    columns = ["EVENT_ID", "NUMBER", "LAP_NUMBER"] + FEATURE_COLUMNS
    df = pd.read_parquet(parquet_path, columns=columns)

    if event_id is not None:
        df = df[df["EVENT_ID"] == event_id]
    if number is not None:
        df = df[df["NUMBER"] == number]
    if event_id is None and number is None and len(df):
        first_event, first_number = df.iloc[0][["EVENT_ID", "NUMBER"]]
        df = df[(df["EVENT_ID"] == first_event) & (df["NUMBER"] == first_number)]

    if df.empty:
        raise ValueError(
            f"load_race_laps: 조건에 맞는 랩 데이터가 없습니다 (event_id={event_id}, number={number})"
        )

    return df.sort_values("LAP_NUMBER").reset_index(drop=True)


class PitstopEnv(gym.Env):
    """
    WEC 제네시스 피트스탑 의사결정 시뮬레이션 환경 (실제 레이스 리플레이 버전)

    race_laps로 받은 실제 차량 한 대의 랩별 기록(순위/격차/날씨 등)을 랩 순서대로
    그대로 재생한다. 에이전트는 매 랩 "피트인 여부"만 결정하고, 그 결정은
    STINT_LAP(타이어 마모 대리 변수)과 보상에 반영된다. 순위/격차 자체는 실제
    레이스 기록을 따라가므로, 여기서 5랩 슬라이딩 윈도우로 계산하는 추세 피처
    (pos_change_5lap 등)도 실제 레이스 흐름을 그대로 반영한다.
    """

    REQUIRED_COLUMNS = ["LAP_NUMBER"] + FEATURE_COLUMNS

    def __init__(self, race_laps: pd.DataFrame, evaluator=None, trend_window: int = TREND_WINDOW_LAPS):
        super().__init__()

        missing = set(self.REQUIRED_COLUMNS) - set(race_laps.columns)
        if missing:
            raise ValueError(f"PitstopEnv: race_laps에 필수 컬럼 누락: {missing}")

        self.race_laps = race_laps.sort_values("LAP_NUMBER").reset_index(drop=True)
        self.lap_numbers = self.race_laps["LAP_NUMBER"].to_numpy()
        self.first_lap = int(self.lap_numbers[0])
        self.max_lap = int(self.lap_numbers[-1])
        self.trend_window = trend_window

        # Action Space: 0 (계속 주행), 1 (피트인)
        self.action_space = spaces.Discrete(2)

        # Observation Space: FEATURE_COLUMNS_TREND(11개) 순서 그대로.
        # [LAP_PROGRESS_RATIO, STINT_LAP, CLASS_POSITION, GAP_TO_LEADER_SEC, GAP_TO_AHEAD_SEC,
        #  WEATHER_CATEGORY, TRACK_TEMP, pos_change_5lap, gap_change_5lap, pos_slope_5lap, gap_slope_5lap]
        low = np.array(
            [0.0, 0, 1, -300.0, -300.0, 0, 0.0, -30.0, -300.0, -30.0, -300.0], dtype=np.float32
        )
        high = np.array(
            [1.0, 40, 30, 300.0, 300.0, 2, 100.0, 30.0, 300.0, 30.0, 300.0], dtype=np.float32
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_lap = self.first_lap
        self.stint_lap = 1

        self._pos_window = SlidingTrendWindow(trend_window)
        self._gap_window = SlidingTrendWindow(trend_window)

        # 평가기 주입 지점: 인자로 안 넘기면 기존처럼 더미 평가기 사용(하위호환).
        # Day3부터는 evaluator/xgb_evaluator.py::load_evaluator()로 실제 XGBoost 모델을 넘긴다.
        self.evaluator = evaluator if evaluator is not None else DummyPodiumEvaluator()
        self.current_podium_prob = 0.0

    def _current_row(self) -> pd.Series:
        # 데이터에 랩 결번이 있을 수 있어(레드플래그 드롭 등) 가장 가까운 이전 랩으로 보정.
        idx = int(np.searchsorted(self.lap_numbers, self.current_lap, side="right")) - 1
        idx = max(0, min(idx, len(self.race_laps) - 1))
        return self.race_laps.iloc[idx]

    def get_obs(self):
        """실제 레이스 랩 데이터를 읽어 11차원 상태(기본 7 + 추세 4)를 반환한다."""
        row = self._current_row()

        self._pos_window.push(row["CLASS_POSITION"])
        self._gap_window.push(row["GAP_TO_LEADER_SEC"])
        pos_change, pos_slope = self._pos_window.change_and_slope()
        gap_change, gap_slope = self._gap_window.change_and_slope()

        return np.array(
            [
                row["LAP_PROGRESS_RATIO"],
                self.stint_lap,           # 실데이터의 STINT_LAP이 아니라 에이전트의 실제 피트 결정을 반영
                row["CLASS_POSITION"],
                row["GAP_TO_LEADER_SEC"],
                row["GAP_TO_AHEAD_SEC"],
                row["WEATHER_CATEGORY"],
                row["TRACK_TEMP"],
                pos_change,
                gap_change,
                pos_slope,
                gap_slope,
            ],
            dtype=np.float32,
        )

    def _get_podium_prob(self, obs):
        """🌟 핵심: numpy 배열을 pandas DataFrame으로 변환하여 평가기에 전달"""
        df = pd.DataFrame([obs], columns=FEATURE_COLUMNS_TREND)
        # predict_proba는 배열을 반환하므로 첫 번째 값([0]) 추출
        return self.evaluator.predict_proba(df)[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_lap = self.first_lap
        self.stint_lap = 1
        self._pos_window.reset()
        self._gap_window.reset()

        obs = self.get_obs()
        self.current_podium_prob = self._get_podium_prob(obs)

        return obs, {}

    def step(self, action):
        if action == 1:
            self.stint_lap = 1  # 피트인
        else:
            self.stint_lap += 1  # 주행

        self.current_lap += 1
        obs = self.get_obs()

        # 보상(Reward) = 확률 변화량 계산
        new_podium_prob = self._get_podium_prob(obs)
        reward = new_podium_prob - self.current_podium_prob
        self.current_podium_prob = new_podium_prob

        terminated = self.current_lap >= self.max_lap
        truncated = False

        return obs, reward, terminated, truncated, {}

    def action_masks(self):
        """팀원 B의 schema.py 상수(MIN_STINT_LAPS, MAX_STINT_LAPS) 활용"""
        masks = [True, True]

        if self.stint_lap < MIN_STINT_LAPS:
            masks[1] = False

        if self.stint_lap >= MAX_STINT_LAPS:
            masks[0] = False

        return masks


# ==========================================
#  통합 테스트 로직
# ==========================================
if __name__ == "__main__":
    race_laps = load_race_laps()
    env = PitstopEnv(race_laps=race_laps)
    obs, _ = env.reset()
    print(f" 초기화 완료! | 초기 확률: {env.current_podium_prob:.2f}")
    print(f"   초기 상태(11차원): {np.round(obs, 2)}")

    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        print(f"[{i+1}랩 주행] 보상: {reward:+.2f} | 갱신된 확률: {env.current_podium_prob:.2f} | 마스크: {env.action_masks()}")
