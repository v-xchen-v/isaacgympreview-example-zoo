from isaacgym import gymapi
from isaacgym import gymtorch
import torch

# Initialize gym
gym = gymapi.acquire_gym()

# Create a simulation environment
sim_params = gymapi.SimParams()
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # Set gravity
sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)
sim_params.use_gpu_pipeline = True

# Create environment
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 2)

# Load URDF model (Replace with actual URDF file)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # Keep the base fixed
asset_options.flip_visual_attachments = True  # Do not flip visual attachments

# add cartpole urdf asset
asset_root = "./assets"
asset_file = "urdf/franka_description/robots/franka_panda.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Create actor
actor_handle = gym.create_actor(env, robot_asset, gymapi.Transform(), "robot", 0, 1)

#---------  Setting Target Position & Velocity for a Specific Actor -------------- #
# Get an actor's first DOF index
env_idx = 0  # Change based on the environment index
actor_handle = gym.get_actor_handle(env, 0)
actor_dof_start = gym.get_actor_dof_index(env, actor_handle, 0, gymapi.DOMAIN_SIM)
num_actor_dofs = gym.get_actor_dof_count(env, actor_handle)

# Get number of total DOFs in the simulation
num_dofs = gym.get_sim_dof_count(sim)

# Create a tensor to hold target velocities (default zeros)
dof_target_vel_tensor = torch.zeros(num_dofs, dtype=torch.float32, device='cpu')
# Create a tensor to hold target positions (default zeros)
dof_target_pos_tensor = torch.zeros(num_dofs, dtype=torch.float32, device='cpu')


# Modify target positions and velocities for this actor
dof_target_pos_tensor[actor_dof_start: actor_dof_start + num_actor_dofs] = 1.0
dof_target_vel_tensor[actor_dof_start: actor_dof_start + num_actor_dofs] = 1.5

# Apply the updated tensors
# set_dof_position_target_tensor and set_dof_velocity_target_tensor take in a tensor of size (num_dofs) in the CPU(not GPU) as input
gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_target_pos_tensor))
gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(dof_target_vel_tensor))

dof_states = gym.acquire_dof_state_tensor(sim)
dof_states_tensor = gymtorch.wrap_tensor(dof_states)
# Step the similation
for i in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    # Print the DoF states tensor in each step
    print(dof_states_tensor)
    
    gym.refresh_dof_state_tensor(sim)
    
# plot the dof states at steps
import matplotlib.pyplot as plt
plt.plot(dof_states_tensor.cpu().numpy())
plt.xlabel('DOF Index')
plt.ylabel('DOF State')
plt.title('DOF States')
plt.show()