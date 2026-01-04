# Ref: 

import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch

# TODO: ?apply set target position at one step or every 10 steps, what's the difference?
# TODO: ?figure out the multiple set target postion apis, the data types of position, conversion between them, can cpu and gpu switchs?
# TODO: ?cpu mode gives out more detailed error informations, how to use that feature to better debug?

# Initialize gym
gym = gymapi.acquire_gym()

# Create a simulation environment
sim_params = gymapi.SimParams()
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # Set gravity
sim = gym.create_sim(0, 0, gymapi.SimType.SIM_PHYSX, sim_params)

# Create environment
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 2)

# Load URDF model (Replace with actual URDF file)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # Keep the base fixed
asset_options.flip_visual_attachments = True  # Do not flip visual attachments

# add cartpole urdf asset
asset_root = "../assets"
asset_file = "urdf/franka_description/robots/franka_panda.urdf"
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Create actor
actor_handle = gym.create_actor(env, robot_asset, gymapi.Transform(), "robot", 0, 1)

# Get DOF count
dof_count = gym.get_asset_dof_count(robot_asset)

# Acquire DOF state tensor
dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(dof_states)

# Acquire DOF position and velocity tensors
dof_positions = dof_states.view(torch.float32)[:, 0]  # Extract positions
dof_velocities = dof_states.view(torch.float32)[:, 1]  # Extract velocities

# Define target pose
target_positions = np.linspace(-0.5, 0.5, dof_count)  # Set some target positions
initial_positions = np.zeros(dof_count)  # Set initial positions

# Create a control action tensor
# why 2? 0 for position, 1 for velocity
dof_control_positions = torch.tensor(target_positions, dtype=torch.float32)
dof_control = torch.zeros(dof_count, 2, dtype=torch.float32, device='cuda')

# Simulation loop
step = 0
# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()
while True:
    if step == 1:
        gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(dof_control_positions))  # Apply control
    
    # Step simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    
    step += 1

    # Reset after 300 steps
    if step >= 300:
        print("Resetting environment...")
        
        # Apply control to move towards target pose
        dof_control[:, 0] = torch.tensor(initial_positions, dtype=torch.float32, device='cuda')  # Target positions
        dof_control[:, 1] = 0.0  # Zero velocity
        # convert to gymapi.DofState.dtype
        # Create a structured numpy array with gymapi.DofState.dtype
        dof_control_np = dof_control.cpu().numpy()
        dof_state_array = np.zeros(dof_control_np.shape[0], dtype=gymapi.DofState.dtype)

        # Assign values
        for i in range(dof_control_np.shape[0]):
            dof_state_array[i]['pos'] = dof_control_np[i, 0]  # Assign position
        
        reset_positions = np.zeros(dof_count)  # Reset all joints to 0 position
        reset_velocities = np.zeros(dof_count)  # Zero out velocities
        
        # Create reset DOF state structure
        dof_state_array = np.zeros(dof_count, dtype=gymapi.DofState.dtype)
        for i in range(dof_count):
            dof_state_array[i]['pos'] = reset_positions[i]
            dof_state_array[i]['vel'] = reset_velocities[i]

        # Apply reset DOF states
        gym.set_actor_dof_states(env, actor_handle, dof_state_array, gymapi.STATE_ALL)
        
        step = 0  # Restart counter

    # Render the scene
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)