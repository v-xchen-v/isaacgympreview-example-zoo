import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        
        # Define observation space (continuous values)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Define action space (continuous action)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Environment state
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0

    def reset(self):
        self.state = np.random.uniform(-0.5, 0.5, size=(4,))
        self.step_count = 0
        return self.state

    def step(self, action):
        self.state += action  # Simple update rule
        reward = -np.sum(np.abs(self.state))  # Reward is negative of sum of absolute values
        self.step_count += 1
        done = self.step_count >= 50  # End episode after 50 steps
        return self.state, reward, done, {}

    def render(self, mode="human"):
        print(f"State: {self.state}")

    def close(self):
        pass

from gym.envs.registration import register

register(
    id="CustomEnv-v0",
    entry_point="custom_env:CustomEnv",
)
