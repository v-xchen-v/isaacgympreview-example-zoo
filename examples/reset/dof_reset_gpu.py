"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates how to reset the robot's DoF states with GPU pipeline.
- acquire the initial dof states tensor and root states tensor
- reset the physics states, in GPU pipeline, for effiency , reset the root states tensor and dof states tensor,
    instead of rigid body states and dof states in CPU mode.

Usage:
cd isaacgympreview-example-zoo
python examples/reset/dof_reset_gpu.py

"""

import math
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = 4

# use gpu_pipeline(use Gym tensor API), and run experiment on CPU, to more detailed information printting
# which is very useful for debugging.
sim_params.physx.use_gpu = False
# the gym tensor's device should align with the sim_params.use_gpu_pipeline
# GUI: information not available when using GPU pipeline
sim_params.use_gpu_pipeline = True 
if not args.use_gpu_pipeline:
    assert("WARNING: Forcing GPU pipeline.")

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 4
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# add cartpole urdf asset
asset_root = "./assets"
asset_file = "urdf/cartpole.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cartpole_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
initial_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole0 = gym.create_actor(env0, cartpole_asset, initial_pose, 'cartpole', 0, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cartpole0)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
props["stiffness"] = (5000.0, 5000.0)
props["damping"] = (100.0, 100.0)
gym.set_actor_dof_properties(env0, cartpole0, props)
# Set DOF drive targets
# TODO: use find_actor_dof_handle to find the handle(numberic index) of the DOF in the actor by name.
cart_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'slider_to_cart') # 0
pole_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'cart_to_pole') # 1
gym.set_dof_target_position(env0, cart_dof_handle0, 0)
gym.set_dof_target_position(env0, pole_dof_handle0, 0.25 * math.pi)

# Create environment 1
# Cart held steady using position target mode.
# Pole rotating using velocity target mode.
env1 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole1 = gym.create_actor(env1, cartpole_asset, initial_pose, 'cartpole', 1, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env1, cartpole1)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL)
props["stiffness"] = (5000.0, 0.0)
props["damping"] = (100.0, 200.0)
gym.set_actor_dof_properties(env1, cartpole1, props)
# Set DOF drive targets
cart_dof_handle1 = gym.find_actor_dof_handle(env1, cartpole1, 'slider_to_cart')
pole_dof_handle1 = gym.find_actor_dof_handle(env1, cartpole1, 'cart_to_pole')
gym.set_dof_target_position(env1, cart_dof_handle1, 0)
gym.set_dof_target_velocity(env1, pole_dof_handle1, -2.0 * math.pi)

# Create environment 2
# Cart moving side to side using velocity target mode.
# Pole held steady using position target mode.
env2 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole2 = gym.create_actor(env2, cartpole_asset, initial_pose, 'cartpole', 2, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env2, cartpole2)
props["driveMode"] = (gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_POS)
props["stiffness"] = (0.0, 5000.0)
props["damping"] = (200.0, 100.0)
gym.set_actor_dof_properties(env2, cartpole2, props)
# Set DOF drive targets
cart_dof_handle2 = gym.find_actor_dof_handle(env2, cartpole2, 'slider_to_cart')
pole_dof_handle2 = gym.find_actor_dof_handle(env2, cartpole2, 'cart_to_pole')
gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)
gym.set_dof_target_position(env2, pole_dof_handle2, 0.0)

# Create environment 3
# Cart has no drive mode, but will be pushed around using forces.
# Pole held steady using position target mode.
env3 = gym.create_env(sim, env_lower, env_upper, 2)
cartpole3 = gym.create_actor(env3, cartpole_asset, initial_pose, 'cartpole', 3, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env3, cartpole3)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_EFFORT)
props["stiffness"] = (5000.0, 0.0)
props["damping"] = (100.0, 0.0)
gym.set_actor_dof_properties(env3, cartpole3, props)
# Set DOF drive targets
cart_dof_handle3 = gym.find_actor_dof_handle(env3, cartpole3, 'slider_to_cart')
pole_dof_handle3 = gym.find_actor_dof_handle(env3, cartpole3, 'cart_to_pole')
gym.set_dof_target_position(env3, cart_dof_handle3, 0.0)
gym.apply_dof_effort(env3, pole_dof_handle3, 200)

# TODO: !MUST call prepare_sim, after all the environments are fully set up, to initial GPU memory used by tensor API!
# error:
# - Failed to fill root state tensor
gym.prepare_sim(sim)

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

step = 0
# Simulate
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Nothing to be done for env 0

    # Nothing to be done for env 1
    # TODO: Function GymSetDofTargetVelocity cannot be used with the GPU pipeline after simulation starts.  Please use the tensor API if possible.  See docs/programming/tensors.html for more info.!
    # TODO: before simulation starts, you can use the function just as off use gpu_pipeline, but after starts, must use Tensor API, to acquire states and set targets.
    # Update env 2: reverse cart target velocity when bounds reached
    # ---------------- Functions of set target with tensor API ----------------
    def get_dof_position(env_handle, actor_handle, dof_index_in_actor):
        # TODO: !MUST call refresh_dof_state_tensor, synchronize the state of the degrees of freedom (DOF) between the simulator and your code. !
        gym.refresh_dof_state_tensor(sim)
        dof_states = gym.acquire_dof_state_tensor(sim)
        dof_states_tensor = gymtorch.wrap_tensor(dof_states)
        actor_dof_start = gym.get_actor_dof_index(env_handle, actor_handle, 0, gymapi.DOMAIN_SIM)
        num_actor_dofs = gym.get_actor_dof_count(env_handle, actor_handle)
        actor_dof_states = dof_states_tensor[actor_dof_start: actor_dof_start + num_actor_dofs, 0]
        return actor_dof_states.cpu().numpy()[dof_index_in_actor]
    
    def set_dof_target_velocity(env_handle, actor_handle, dof_index_in_actor, target_velocity):
        dof_states = gym.acquire_dof_state_tensor(sim)
        dof_states_tensor = gymtorch.wrap_tensor(dof_states) # [num_sim_dofs, 2]
        target_velocities_tensor = dof_states_tensor[:, 1].clone()
        actor_dof_start = gym.get_actor_dof_index(env_handle, actor_handle, dof_index_in_actor, gymapi.DOMAIN_SIM)
        target_velocities_tensor[actor_dof_start] = target_velocity
        gym.set_dof_velocity_target_tensor(sim, gymtorch.unwrap_tensor(target_velocities_tensor))
        
    def set_dof_target_velocity_v2(env_handle, actor_handle, dof_index_in_actor, target_velocity):
        dof_states = gym.acquire_dof_state_tensor(sim)
        dof_states_tensor = gymtorch.wrap_tensor(dof_states) # [num_sim_dofs, 2]
        target_velocities_tensor = dof_states_tensor[:, 1].clone()
        actor_dof_start = gym.get_actor_dof_index(env_handle, actor_handle, dof_index_in_actor, gymapi.DOMAIN_SIM)
        actor_index = gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
        target_velocities_tensor[actor_dof_start] = target_velocity
        actor_indices = torch.tensor([actor_index], dtype=torch.int32, device='cuda')
        gym.set_dof_velocity_target_tensor_indexed(sim, gymtorch.unwrap_tensor(target_velocities_tensor),
                                                   gymtorch.unwrap_tensor(actor_indices), len(actor_indices))
        
    def apply_dof_effort(env_handle, actor_handle, dof_index_in_actor, effort):
        dof_efforts_tensor = torch.zeros(gym.get_sim_dof_count(sim), dtype=torch.float32, device='cuda')
        actor_dof_start = gym.get_actor_dof_index(env_handle, actor_handle, dof_index_in_actor, gymapi.DOMAIN_SIM)
        dof_efforts_tensor[actor_dof_start] = effort
        ret = gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(dof_efforts_tensor))
        # print(f'Successfully apply efforts: {ret}')
        
    def apply_dof_effort_v2(env_handle, actor_handle, dof_index_in_actor, effort):
        dof_efforts_tensor = torch.zeros(gym.get_sim_dof_count(sim), dtype=torch.float32, device='cuda')
        actor_index = gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
        actor_dof_start = gym.get_actor_dof_index(env_handle, actor_handle, dof_index_in_actor, gymapi.DOMAIN_SIM)
        dof_efforts_tensor[actor_dof_start] = effort
        actor_indices = torch.tensor([actor_index], dtype=torch.int32, device='cuda')
        
        # set_xxx_tensor_indexed compared to set_xxx_tensor, is just a masked version of writing back to the global tensor.
        ret = gym.set_dof_actuation_force_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_efforts_tensor), 
                                                         gymtorch.unwrap_tensor(actor_indices),
                                                         len(actor_indices))
        # print(f'Successfully apply efforts: {ret}')
                                                  
        
    pos = get_dof_position(env2, cartpole2, cart_dof_handle2)
    if pos >= 0.5:
        # gym.set_dof_target_velocity(env2, cart_dof_handle2, -1.0)
        set_dof_target_velocity(env2, cartpole2, cart_dof_handle2, -1.0)
    elif pos <= -0.5:
        # gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)
        set_dof_target_velocity_v2(env2, cartpole2, cart_dof_handle2, 1.0)

    # Update env 3: apply an effort to the pole to keep it upright
    pos = get_dof_position(env3, cartpole3, pole_dof_handle3)
    # gym.apply_dof_effort(env3, pole_dof_handle3, -pos * 50)
    apply_dof_effort_v2(env3, cartpole3, pole_dof_handle3, -pos * 500)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    
    if step == 0:
        # Store the initial DoF states
        # cart0_dof_states = gym.get_actor_dof_states(env0, cartpole0, gymapi.STATE_ALL)
        # cart1_dof_states = gym.get_actor_dof_states(env1, cartpole1, gymapi.STATE_ALL)
        # cart2_dof_states = gym.get_actor_dof_states(env2, cartpole2, gymapi.STATE_ALL)
        # cart3_dof_states = gym.get_actor_dof_states(env3, cartpole3, gymapi.STATE_ALL)
        _global_dof_states = gym.acquire_dof_state_tensor(sim)
        global_dof_states = gymtorch.wrap_tensor(_global_dof_states).clone()
        
        # No inital DoF targets, store initial position instead

        
        # initial_body_states = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL) # [num_rigid_bodies, 3+4+3+3], position(3)+quat(4)+linear_vel(3)+angular_vel(4)
        # TODO: MUST call refresh_actor_root_state_tensor, synchronize the state of the root bodies between the simulator and your code.
        gym.refresh_actor_root_state_tensor(sim) 
        # if acquire_actor_root_state_tensor got all zeros data, means it either not initialized or not properly refreshed.
        _root_states = gym.acquire_actor_root_state_tensor(sim) # [num_actor, 3+4+3+4], num_envs*num_actor_per_env, position(3)+quat(4)+linear_vel(3)+angular_vel(4)
        # clone the initial states tensor, to avoid later refresh the tensor, the tensor will be changed.
        root_states = gymtorch.wrap_tensor(_root_states).clone()
        pass
        pass

    def reset():
        print("Resetting environment...")
        # Reset physics states, includes rigid body states and dof states
        # Presently, the rigid body state tensor is read-only. Setting rigid body states is only allowed for actor root bodies using the root state tensor.
        
        # Reset the actor's transform
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root_states))
        
        # # Reset the DOF states
        # gym.set_actor_dof_states(env0, cartpole0, cart0_dof_states, gymapi.STATE_ALL)
        # gym.set_actor_dof_states(env1, cartpole1, cart1_dof_states, gymapi.STATE_ALL)
        # gym.set_actor_dof_states(env2, cartpole2, cart2_dof_states, gymapi.STATE_ALL)
        # gym.set_actor_dof_states(env3, cartpole3, cart3_dof_states, gymapi.STATE_ALL)
        gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(global_dof_states))
        
        # Reset the DoF target position, in this example different DoF use different drive mode, we ignore it here, actually you can use the 
        # set_dof_xxx_target_tensor to reset it.
        # gym.set_dof_position_target_tensor_indexed(...)

    if step % 100 == 0:
        reset()
        step = 0
    
    step += 1

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
