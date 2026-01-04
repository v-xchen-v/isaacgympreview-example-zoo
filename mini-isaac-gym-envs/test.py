import gymnasium as gym
from isaacgym import gymapi
import torch

class VecTaskWrapper(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.task = VecTask(config)  # Wrap the VecTask environment
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self):
        obs = self.task.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done = self.task.step(action)
        return obs, reward, done, False, {}

# Register environment with rl_games
from rl_games.common import env_configurations
env_configurations.register("VecTaskEnv", {"env_creator": lambda **kwargs: VecTaskWrapper(kwargs)})

# Run training
import os
os.system("python -m rl_games.train --config=ppo_config.yaml")
