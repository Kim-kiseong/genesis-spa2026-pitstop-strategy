import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PitstopEnv(gym.Env):
    """
    WEC 제네시스 피트스탑 의사결정 시뮬레이션 환경 (Gymnasium 규격)
    """
    def __init__(self):
        super(PitstopEnv, self).__init__()
        
        # 1. Action Space: 0 (계속 주행), 1 (피트인)
        self.action_space = spaces.Discrete(2)
        
        # 2. Observation Space: [stint_lap, track_temp, gap_to_leader_sec, weather_category]
        # (임시 범위 지정: Low~High)
        low = np.array([0, 0.0, -300.0, 0], dtype=np.float32)
        high = np.array([40, 100.0, 300.0, 2], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.current_lap = 1
        self.stint_lap = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_lap = 1
        self.stint_lap = 1
        
        # 초기 상태 반환 (임시 데이터)
        obs = np.array([self.stint_lap, 30.0, 0.0, 0], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        # 0: 주행, 1: 피트스탑
        if action == 1:
            self.stint_lap = 1  # 피트인 시 stint_lap 리셋
        else:
            self.stint_lap += 1
            
        self.current_lap += 1
        
        # 보상(Reward) 및 종료 조건 설정 (Day 1에는 더미 보상)
        reward = 0.1
        terminated = self.current_lap >= 100  # 100랩 달성 시 에피소드 종료
        truncated = False
        
        obs = np.array([self.stint_lap, 30.0, 0.0, 0], dtype=np.float32)
        info = {}
        
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    # 환경 동작 테스트
    env = PitstopEnv()
    obs, _ = env.reset()
    print("✅ 환경 생성 및 reset 성공! 초기 상태:", obs)
    
    # 1스텝 동작 테스트 (Action 0: 계속 주행)
    obs, reward, terminated, truncated, info = env.step(0)
    print("✅ step(0) 실행 성공! 다음 상태:", obs, "보상:", reward)