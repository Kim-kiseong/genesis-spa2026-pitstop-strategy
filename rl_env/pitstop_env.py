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

        # ... 기존 step 함수 코드 ...

    def action_masks(self):
        """
        MaskablePPO가 매 랩마다 호출하여 '현재 가능한 행동'이 무엇인지 확인하는 함수.
        반환값: [계속 주행 가능 여부(True/False), 피트인 가능 여부(True/False)]
        """
        masks = [True, True] # 기본적으로 주행(0), 피트인(1) 둘 다 가능하다고 가정
        
        # 규칙 1: 조기 피트스탑 방지 (stint_lap이 3 미만일 때는 피트인 불가)
        if self.stint_lap < 3:
            masks[1] = False
            
        # 규칙 2: 연료 고갈 강제 피트인 (stint_lap이 34 이상 도달 시 계속 주행 불가)
        if self.stint_lap >= 34:
            masks[0] = False
            
        return masks

if __name__ == "__main__":
    env = PitstopEnv()
    obs, _ = env.reset()
    print("✅ 초기 상태:", obs, "| 현재 가능 행동(Mask):", env.action_masks())
    
    # 1스텝 동작 테스트 (Action 0: 계속 주행)
    obs, reward, terminated, truncated, info = env.step(0)
    print("✅ 1랩 주행 후 상태:", obs, "| 현재 가능 행동(Mask):", env.action_masks())

    # 강제로 34랩으로 만들어보기 (규칙 2 테스트)
    env.stint_lap = 34
    print("🚨 연료 고갈 임박(34랩) 시 가능 행동(Mask):", env.action_masks())