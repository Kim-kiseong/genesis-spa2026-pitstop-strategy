import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

# 팀원 B가 작성한 evaluator 모듈 임포트
from evaluator.schema import FEATURE_COLUMNS, MIN_STINT_LAPS, MAX_STINT_LAPS
from evaluator.dummy_evaluator import DummyPodiumEvaluator

class PitstopEnv(gym.Env):
    """
    WEC 제네시스 피트스탑 의사결정 시뮬레이션 환경 (XGBoost 연동 버전)
    """
    def __init__(self, evaluator=None):
        super(PitstopEnv, self).__init__()

        # Action Space: 0 (계속 주행), 1 (피트인)
        self.action_space = spaces.Discrete(2)

        # Observation Space: 팀원 B의 FEATURE_COLUMNS (7개) 기준
        # ["LAP_PROGRESS_RATIO", "STINT_LAP", "CLASS_POSITION", "GAP_TO_LEADER_SEC", "GAP_TO_AHEAD_SEC", "WEATHER_CATEGORY", "TRACK_TEMP"]
        low = np.array([0.0, 0, 1, -300.0, -300.0, 0, 0.0], dtype=np.float32)
        high = np.array([1.0, 40, 30, 300.0, 300.0, 2, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_lap = 1
        self.stint_lap = 1

        # 평가기 주입 지점: 인자로 안 넘기면 기존처럼 더미 평가기 사용(하위호환).
        # Day3부터는 evaluator/xgb_evaluator.py::load_evaluator()로 실제 XGBoost 모델을 넘긴다.
        self.evaluator = evaluator if evaluator is not None else DummyPodiumEvaluator()
        self.current_podium_prob = 0.0

    def get_obs(self):
        """현재 환경의 상태를 7개 차원의 배열로 반환 (임시 더미 값 포함)"""
        lap_progress = min(self.current_lap / 100.0, 1.0)
        class_position = 1.0
        gap_to_leader = 0.0
        gap_to_ahead = 0.0
        weather_category = 0.0
        track_temp = 30.0
        
        return np.array([
            lap_progress,       # LAP_PROGRESS_RATIO
            self.stint_lap,     # STINT_LAP
            class_position,     # CLASS_POSITION
            gap_to_leader,      # GAP_TO_LEADER_SEC
            gap_to_ahead,       # GAP_TO_AHEAD_SEC
            weather_category,   # WEATHER_CATEGORY
            track_temp          # TRACK_TEMP
        ], dtype=np.float32)

    def _get_podium_prob(self, obs):
        """🌟 핵심: numpy 배열을 pandas DataFrame으로 변환하여 평가기에 전달"""
        df = pd.DataFrame([obs], columns=FEATURE_COLUMNS)
        # predict_proba는 배열을 반환하므로 첫 번째 값([0]) 추출
        return self.evaluator.predict_proba(df)[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_lap = 1
        self.stint_lap = 1
        
        obs = self.get_obs()
        self.current_podium_prob = self._get_podium_prob(obs)
        
        return obs, {}

    def step(self, action):
        if action == 1:
            self.stint_lap = 1  # 피트인
        else:
            self.stint_lap += 1 # 주행
            
        self.current_lap += 1
        obs = self.get_obs()
        
        # 보상(Reward) = 확률 변화량 계산
        new_podium_prob = self._get_podium_prob(obs)
        reward = new_podium_prob - self.current_podium_prob
        self.current_podium_prob = new_podium_prob
        
        terminated = self.current_lap >= 100
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
    env = PitstopEnv()
    obs, _ = env.reset()
    print(f" 초기화 완료! | 초기 확률: {env.current_podium_prob:.2f}")
    print(f"   초기 상태(7차원): {np.round(obs, 2)}")
    
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        print(f"[{i+1}랩 주행] 보상: {reward:+.2f} | 갱신된 확률: {env.current_podium_prob:.2f} | 마스크: {env.action_masks()}")