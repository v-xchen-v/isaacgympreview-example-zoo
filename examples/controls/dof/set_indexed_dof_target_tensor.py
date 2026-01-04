"""The global DoF target position tensor in Isaac Gym has a shape of [total_num_dofs]. where
- total_num_dofs = num_actors * num_dofs_per_actor (sum of all DoFs across all actors)

Only have set_actor_dof_position_target_tensor, no get.
    
The example below demonstrates how to set global target position tensor in Isaac Gym's GPU pipeline, follow these steps:
1. Create a 


Summary:
1. Isaac gym does NOT provide acquire_dof_target_position_tensor()
2. Create your own tensor (torch.zeros(num_dofs, device='cuda')) to store target positions and update it.
3. Use gym.set_dof_position_target_tensor(sim, target_positions) to apply the changes back to the simulation.
4. Find an actor's DoF range using gym.get_actor_dof_index() and modify only relevant indices.
"""

from isaacgym import gymapi, gymtorch
import torch

# Initialize gym
gym = gymapi.acquire_gym()

# Create a simulation environment
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Load cartpole URDF asset
asset_root = "./assets"
asset_file = "urdf/cartpole.urdf"
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Create an actor
initial_pose = gymapi.Transform()
num_envs = 2
envs = []
# create two environments
for i in range(num_envs):
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
    envs.append(env)
    actor_handle = gym.create_actor(env, cartpole_asset, initial_pose, "actor", 0, 1)

gym.prepare_sim(sim)

# Acquire the global DoF state tensor
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)

# Can not Acquire the global DoF target position tensor
# dof_target_position_tensor = gym.acquire_dof_position_target_tensor(sim)
# dof_target_position = gymtorch.wrap_tensor(dof_target_position_tensor) # Convert to PyTorch tensor

# Check the shape (should be [total_dofs])
# print("Global DoF target position tensor shape:", dof_target_position.shape)

# Check shape
print("Global DoF state tensor shape:", dof_state.shape)
# Expected output: torch.Size([total_dofs, 2])

# Get an actor's first DoF index
env_idx = 0
actor_handle = gym.get_actor_handle(
    envs[env_idx], 0
)  # locate actor handle by (env_handle, actor_index)
actor_dof_start = gym.get_actor_dof_index(
    envs[env_idx], actor_handle, 0, gymapi.DOMAIN_SIM
)  # First DoF index
num_actor_dofs = gym.get_actor_dof_count(envs[env_idx], actor_handle)  # DoFs per actor
actor_index = gym.get_actor_index(envs[env_idx], actor_handle, gymapi.DOMAIN_SIM)

# Extract DoF states for this actor
target_dof_state = dof_state.clone()  # Clone the DoF state tensor
actor_dof_states = target_dof_state[
    actor_dof_start : actor_dof_start + num_actor_dofs, :
]

print(f"Actor 0 DoF States (Pos, Vel): \n{actor_dof_states}")

# Modify the first DoF of the actor
actor_dof_states[0, 0] = 1.2  # New position
target_position = target_dof_state[:, 0].contiguous()
actor_dof_states[0, 1] = 0.5  # New velocity
target_velocity = target_dof_state[:, 1].contiguous()

# Apply the changes back to the simulation
# actor_indices tensor must be intialize outside of gymtorch.unwrap_tensor
# actor_indices = torch.tensor([0, 1], dtype=torch.int32)
# ret = gym.set_dof_position_target_tensor_indexed(
#     sim,
#     gymtorch.unwrap_tensor(target_position),
#     gymtorch.unwrap_tensor(actor_indices),
#     2,
# )
# # ret = gym.set_dof_position_target_tensor_indexed(
# #     sim,
# #     gymtorch.unwrap_tensor(target_position),
# #     gymtorch.unwrap_tensor(torch.tensor([0, 1], dtype=torch.int32)), # lead to error
# #     2,
# # )
# print(ret)
# ret = gym.set_dof_velocity_target_tensor_indexed(
#     sim,
#     gymtorch.unwrap_tensor(target_velocity),
#     gymtorch.unwrap_tensor(actor_indices),
#     2,
# )
# print(ret)

# Notices:
# 1. actor_indices tensor must be intialize outside of gymtorch.unwrap_tensor
# 2. actor_indices tensor must be of type torch.int32
# 3. actor_indices tensor must be of size (num_actors)
# 4. The target_position and target_velocity tensors must be of size (num_dofs)
# 5. The target_position and target_velocity tensors must be contiguous
# 6. The target_position and target_velocity tensors must be cpu tensors
actor_indices = torch.tensor([0], dtype=torch.int32)
ret = gym.set_dof_position_target_tensor_indexed(sim,
                                           gymtorch.unwrap_tensor(target_position),
                                           gymtorch.unwrap_tensor(actor_indices),
                                           1)
print(f'Set position target is successful: {ret}')
ret = gym.set_dof_velocity_target_tensor_indexed(sim, gymtorch.unwrap_tensor(target_velocity),
                                           gymtorch.unwrap_tensor(actor_indices),
                                           1)
print(f'Set velocity target is successful: {ret}')

# Step the simulation
for _ in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Print the DoF states tensor in each step
    print(dof_state)

    gym.refresh_dof_state_tensor(sim)
