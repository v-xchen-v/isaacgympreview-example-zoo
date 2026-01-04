# from isaacgym import gymapi
# from isaacgym.torch_utils import *
# import numpy as np
# import torch
# import gymnasium as gym

# class MyVecTask(gym.Env):
#     def __init__(self, num_envs=4, device='cuda:0'):
#         self.num_envs = num_envs
#         self.device = device
        
#         # Initialize gym
#         self.gym = gymapi.acquire_gym()
#         self.sim = self._create_sim()
#         self.envs, self.actor_handles = self._create_envs()
        
#         # Buffers or batched data
#         self.num_obs = 4
#         self.obs_buf = torch.zeros((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device)
#         self.rew_buf = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
#         self.done_buf = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device)
        
#     def _create_sim(self):
#         """Create simulation with physics settings"""
#         sim_params = gymapi.SimParams()
#         sim_params.dt = 1.0/60.0
#         sim_params.substeps = 2
#         sim_params.up_axis = gymapi.UP_AXIS_Z
#         sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
#         sim_params.use_gpu_pipeline = True # Enable GPU pipeline
        
#         sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
#         return sim
    
#     def _create_envs(self):
#         """Create multiple environments"""
#         envs = []
#         actor_handles = []
        
#         lower = gymapi.Vec3(-1, -1, 0.0)
#         upper = gymapi.Vec3(1, 1, 1)
        
#         for i in range(self.num_envs):
#             env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
#             asset_options = gymapi.AssetOptions()
#             asset_options.fix_base_link = True
#             asset_root = "./assets"
#             asset = self.gym.load_asset(self.sim, asset_root, "urdf/cartpole.urdf", asset_options)
            
#             actor_handle = self.gym.create_actor(env, asset, gymapi.Transform(), "cartpole", i, 0)
#             actor_handles.append(actor_handle)
#             envs.append(env)
            
#         return envs, actor_handles          
    
#     def reset(self):
#         """Reset all environment"""
#         self.obs_buf = torch.rand((self.num_envs, self.num_obs), dtype=torch.float32, device=self.device) # Example observation
#         self.done_buf[:] = False
        
#         return self.obs_buf, self.rew_buf, self.done_buf
    
#     def step(self, action):
#         """Taking actions, compute reward, and next state """
#         self.obs_buf += torch.randn_like(self.obs_buf) * 0.1  # Example state update
#         self.rew_buf = torch.sum(self.obs_buf, dim=1)  # Example reward function
#         self.done_buf = torch.rand(self.num_envs, device=self.device)
#         return self.obs_buf, self.rew_buf, self.done_buf
    
#     def render(self, mode="human"):
#         pass

#     def close(self):
#         pass
# # Run VecTask Environment
# if __name__ == "__main__":
#     env = MyVecTask(num_envs=4)
#     obs = env.reset()
    
#     for i in range(10):
#         actions = torch.rand((env.num_envs, 1), dtype=torch.float32, device=env.device)
#         obs, rews, dones = env.step(actions)
#         print(f"Step {i}: Reward {rews.cpu().numpy()}, Dones: {dones.cpu().numpy()}, Obs: {obs.cpu().numpy()}")

# import gymnasium as gym
# import numpy as np

# class MyVecTask(gym.Env):  # ✅ Must inherit gym.Env
#     def __init__(self, num_envs=4):
#         super().__init__()
#         self.num_envs = num_envs
#         # self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
#         # self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
#         self.envs = gym.vector.make("CartPole-v1", num_envs=num_envs)

#         # Get observation and action space from the vectorized env
#         self.observation_space = self.envs.single_observation_space
#         self.action_space = self.envs.single_action_space
        
#         self.actions_num = self.action_space.n
        
#     def reset(self):
#         obs = np.random.uniform(-1, 1, size=(self.num_joints * 2,))
#         return obs, {}

#     def step(self, action):
#         obs = np.random.uniform(-1, 1, size=(self.num_joints * 2,))
#         reward = np.random.uniform(-1, 1)
#         done = np.random.choice([True, False])
#         return obs, reward, done, False, {}
