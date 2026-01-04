'''This example demonstates how to get an actor's DoF state tensor in Isaac Gym's GPU pipeline, follow these steps:
1. Acquire the global DoF state tensor.
    Isaac Gym does not store per-actor DoF states in the GPU pipeline. Instead, it stores all DoF states in a global tensor,
    tensors for all actors. To extract an individual actor's DoF states, you need to find the actor's DoF index range within this global tensor.
2. Get an Actor's DoF index range.
    Use the `gym.get_actor_dof_index` function to get the index range of an actor's DoFs in the global DoF state tensor.
3. Extract the actor's DoF states.
4. Optionally, Modify the actor's DoF states.
'''
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
# env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
# actor_handle = gym.create_actor(env, cartpole_asset, initial_pose, "actor", 0, 1)

# Acquire the global DoF state tensor
dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)  # Convert to PyTorch tensor
# DoF state is a (num_dofs, 2) tensor, where the first column is the DoF positions and the second column is the DoF velocities

# DoF state is a (num_dofs, 2) tensor
dof_positions = dof_state[:, 0]  # DoF positions
dof_velocities = dof_state[:, 1]  # DoF velocities

# Get number of DoFs per actor
num_actors = gym.get_actor_count(env)  # Number of actors in the simulation

# Iterate over actors and get their DoF ranges
for env_idx in range(num_envs):
    env = envs[env_idx]
    for actor_idx in range(num_actors):
        # Get the first actor in the environment
        actor_handle = gym.get_actor_handle(env, actor_idx)

        # Get the index range of the actor's DoFs
        dof_index = gym.get_actor_dof_index(env, actor_handle, 0, gymapi.DOMAIN_SIM)  # First DoF index
        num_dofs = gym.get_actor_dof_count(env, actor_handle)  # Number of DoFs

        # Extract the actor's DoF states
        actor_dof_state = dof_state[dof_index: dof_index + num_dofs, :]
        print(f"Env {env_idx} Actor {actor_idx}: DoF States (Pos, Vel): \n", actor_dof_state)
        
        # Optionally, modify the actor's DoF states
        dof_to_modify = 0  # DoF index to modify
        actor_dof_state[dof_to_modify, 0] = 1.0  # Set all DoF positions to 1.0
        actor_dof_state[dof_to_modify, 1] = 0.1  # Set all DoF velocities to 0.1

        # Write the modified DoF states back to the global tensor
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_state))
        
        # Print the modified DoF states
        print(f"Env {env_idx} Actor {actor_idx}: Modified DoF States (Pos, Vel): \n", actor_dof_state)
