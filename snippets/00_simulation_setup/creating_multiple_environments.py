import isaacgym
from isaacgym import gymapi

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

num_envs = 10
lower_bound = gymapi.Vec3(-10, 0, -10)
upper_bound = gymapi.Vec3(10, 0, 10)
for i in range(num_envs):
    env = gym.create_env(sim, lower_bound, upper_bound, num_envs)
    print("Created env: ", env)