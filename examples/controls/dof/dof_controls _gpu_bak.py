"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
Compare to cpu mode, gpu mode use global tensor updating and write APIs to control dofs.

Usage:
cd isaacgympreview-example-zoo
python examples/controls/dof_controls_gpu.py
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
sim_params.physx.use_gpu = True

sim_params.use_gpu_pipeline = True
if args.use_gpu_pipeline:
    print("Using GPU pipeline.")

sim = gym.create_sim(
    0, 0, gymapi.SIM_PHYSX, sim_params
)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError("*** Failed to create viewer")

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

# Create environments 0, 1, 2, 3
envs = []
cartpoles = []
for env_idx in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, 2)
    envs.append(env)
    
    '''[Warning] [carb.gym.plugin] Incorrect actor creation order.
    All actors must be added to an env before creating the next env.
    Adding actors to envs out of order can result in errors.

    To suppress this warning, set enable_actor_creation_warning to False in SimParams.'''
    cartpole = gym.create_actor(env, cartpole_asset, initial_pose, "cartpole", 0, 1)
    cartpoles.append(cartpole)
    
env0 = envs[0]
env1 = envs[1]
env2 = envs[2]
env3 = envs[3]

cartpole0 = cartpoles[0]
cartpole1 = cartpoles[1]
cartpole2 = cartpoles[2]
cartpole3 = cartpoles[3]


# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
# env0 = gym.create_env(sim, env_lower, env_upper, 2)

# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cartpole0)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_POS)
props["stiffness"] = (5000.0, 5000.0)
props["damping"] = (100.0, 100.0)
gym.set_actor_dof_properties(env0, cartpole0, props)

# Set DOF drive targets
# Change the following code to gpu mode:
# ```
# cart_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'slider_to_cart')
# pole_dof_handle0 = gym.find_actor_dof_handle(env0, cartpole0, 'cart_to_pole')
# gym.set_dof_target_position(env0, cart_dof_handle0, 0)
# gym.set_dof_target_position(env0, pole_dof_handle0, 0.25 * math.pi)
# ```
# Ref: https://docs.robotsfan.com/isaacgym/programming/tensors.html#degrees-of-freedom
# API Ref: https://docs.robotsfan.com/isaacgym/api/python/gym_py.html?highlight=find_actor_index#isaacgym.gymapi.Gym.find_actor_index, and seems the third parameter must be gymapi.DOMAIN_ENV
# locate the actor by (env_handle, actor_name) and get the index
global_dof_states = gym.acquire_dof_state_tensor(sim)
cart_num_dofs = gym.get_actor_dof_count(env0, cartpole0)  # Number of DoFs per robot

cart0_index = gym.find_actor_index(env0, "cartpole", gymapi.DOMAIN_ENV)
cart0_first_dof_index = cart0_index * cart_num_dofs
# global_target_position_tensor = torch.zeros(
#     num_envs * cart_num_dofs, dtype=torch.float32, device="cpu"
# )

# meet a case: create a env and an actor inside it, so that the dof_states acquired by 
# acquire_dof_states_tensor is shape of (2, 2), 
# - first dimension: total dof number with num_envs:1, num_actors:1, num_dofs_per_action:2
# - second dimension: 2, position and velocity
# and if more envs and actors are created, the first dimension will be increased.
# so that create all envs and actors af very first may be a good idea.

# use dof_states by acquired makes you do have to care about the actor spawned in sim, only care 
# the env and active in side it, get the index by the (env_handle, actor_name), and modify and write back.

dof_states = gym.acquire_dof_state_tensor(sim) # [num_sim_dofs, 2]
dof_states_tensor = gymtorch.wrap_tensor(dof_states)
dof_states_position_tensor = dof_states_tensor[:, 0].clone()
# dof_states_position_tensor[
#     cart0_first_dof_index : cart0_first_dof_index + cart_num_dofs
# ] = torch.tensor([0.0, 0.25 * math.pi], device=dof_states_position_tensor.device)

# should be contiguous
dof_states_position_tensor = dof_states_position_tensor.contiguous().cpu()

# the tensor should be a indiced position target tensor
actor_indices = torch.tensor([cart0_index]).to(torch.int32)
ret = gym.set_dof_position_target_tensor_indexed(
    sim,
    gymtorch.unwrap_tensor(dof_states_position_tensor),
    gymtorch.unwrap_tensor(actor_indices),
    len([cart0_index])
)  # gpu mode

# Create environment 1
# Cart held steady using position target mode.
# Pole rotating using velocity target mode.
# env1 = gym.create_env(sim, env_lower, env_upper, 2)
# cartpole1 = gym.create_actor(env1, cartpole_asset, initial_pose, "cartpole", 1, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env1, cartpole1)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_VEL)
props["stiffness"] = (5000.0, 0.0)
props["damping"] = (100.0, 200.0)
gym.set_actor_dof_properties(env1, cartpole1, props)
# Set DOF drive targets
# Change the following code to gpu mode:
# ```
# cart_dof_handle1 = gym.find_actor_dof_handle(env1, cartpole1, 'slider_to_cart')
# pole_dof_handle1 = gym.find_actor_dof_handle(env1, cartpole1, 'cart_to_pole')
# gym.set_dof_target_position(env1, cart_dof_handle1, 0)
# gym.set_dof_target_velocity(env1, pole_dof_handle1, -2.0 * math.pi)
# ```
# Acquire the global DoF state tensor, modify the actor's DoF states then write back to the global tensor
dof_states = gym.acquire_dof_state_tensor(sim) # [4, 2]
dof_states_tensor = gymtorch.wrap_tensor(dof_states)
dof_states_position_tensor = dof_states_tensor[:, 0]
dof_states_velocity_tensor = dof_states_tensor[:, 1]

cart1_index = gym.find_actor_index(env1, "cartpole", gymapi.DOMAIN_ENV)
cart1_first_dof_index = cart1_index * cart_num_dofs

# set the position and velocity target tensor
dof_states_position_tensor[cart0_first_dof_index] = 0
dof_states_velocity_tensor[cart0_first_dof_index+1] = -2.0 * math.pi

# gym.get_actor_index(env1, cartpole2, gymapi.DOMAIN_ENV)
# cart_num_dofs = gym.get_actor_dof_count(env1, cartpole1)  # Number of DoFs
# # cart1_dof_index = gym.get_actor_dof_index(env1, cartpole1, 0, gymapi.DOMAIN_SIM)  # First DoF index
# # cart1_dof_states = dof_states[cart1_dof_index: cart1_dof_index + cart1_num_dofs, :]
# cart1_dof_target_pos_tensor = torch.zeros(
#     cart_num_dofs, dtype=torch.float32, device="cpu"
# )
# dof_index = 0
# cart1_dof_target_pos_tensor[dof_index] = 0.0
# cart1_dof_target_vel_tensor = torch.zeros(
#     cart_num_dofs, dtype=torch.float32, device="cpu"
# )
# dof_index = 1
# cart1_dof_target_vel_tensor[dof_index] = -2.0 * math.pi
gym.set_dof_position_target_tensor(
    sim, gymtorch.unwrap_tensor(dof_states_position_tensor.cpu())
)
gym.set_dof_velocity_target_tensor(
    sim, gymtorch.unwrap_tensor(dof_states_velocity_tensor.cpu())
)

# cart1_index = gym.find_actor_index(env1, 'cartpole', gymapi.DOMAIN_ENV)
# # acquire the dof state tensor
# dof_states = gym.acquire_dof_state_tensor(sim)
# dof_states = gymtorch.wrap_tensor(dof_states)
# # get the position tensor

# gym.set_dof_position_target_tensor_indexed(sim,
#                                         gymtorch.unwrap_tensor(torch.tensor([0, -2.0 * math.pi])),
#                                         gymtorch.unwrap_tensor(torch.tensor([cart1_index]).to(torch.int32)),
#                                         len([cart1_index])) # gpu mode

# Create environment 2
# Cart moving side to side using velocity target mode.
# Pole held steady using position target mode.
# env2 = gym.create_env(sim, env_lower, env_upper, 2)
# cartpole2 = gym.create_actor(env2, cartpole_asset, initial_pose, "cartpole", 2, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env2, cartpole2)
props["driveMode"] = (gymapi.DOF_MODE_VEL, gymapi.DOF_MODE_POS)
props["stiffness"] = (0.0, 5000.0)
props["damping"] = (200.0, 100.0)
gym.set_actor_dof_properties(env2, cartpole2, props)
# Set DOF drive targets
# Change the following code to gpu mode:
# ```
# cart_dof_handle2 = gym.find_actor_dof_handle(env2, cartpole2, 'slider_to_cart')
# pole_dof_handle2 = gym.find_actor_dof_handle(env2, cartpole2, 'cart_to_pole')
# gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)
# gym.set_dof_target_position(env2, pole_dof_handle2, 0.0)
# ```
cart2_index = gym.find_actor_index(env2, "cartpole", gymapi.DOMAIN_ENV)
cart2_target_position_tensor = torch.zeros(
    cart_num_dofs, dtype=torch.float32, device="cpu"
)
cart2_target_velocity_tensor = torch.zeros(
    cart_num_dofs, dtype=torch.float32, device="cpu"
)
gym.set_dof_target_velocity_tensor[0] = 1.0
gym.set_dof_target_position_tensor[1] = 0.0
gym.set_dof_position_target_tensor_indexed(
    sim,
    gymtorch.unwrap_tensor(cart2_target_position_tensor),
    gymtorch.unwrap_tensor(torch.tensor([cart2_index]).to(torch.int32)),
    len([cart2_index]),
)
gym.set_dof_velocity_target_tensor_indexed(
    sim,
    gymtorch.unwrap_tensor(cart2_target_velocity_tensor),
    gymtorch.unwrap_tensor(torch.tensor([cart2_index]).to(torch.int32)),
    len([cart2_index]),
)

# Create environment 3
# Cart has no drive mode, but will be pushed around using forces.
# Pole held steady using position target mode.
# env3 = gym.create_env(sim, env_lower, env_upper, 2)
# cartpole3 = gym.create_actor(env3, cartpole_asset, initial_pose, "cartpole", 3, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env3, cartpole3)
props["driveMode"] = (gymapi.DOF_MODE_POS, gymapi.DOF_MODE_EFFORT)
props["stiffness"] = (5000.0, 0.0)
props["damping"] = (100.0, 0.0)
gym.set_actor_dof_properties(env3, cartpole3, props)
# Set DOF drive targets
cart_dof_handle3 = gym.find_actor_dof_handle(env3, cartpole3, "slider_to_cart")
pole_dof_handle3 = gym.find_actor_dof_handle(env3, cartpole3, "cart_to_pole")
# Change the following code to gpu mode:
# ```
# gym.set_dof_target_position(env3, cart_dof_handle3, 0.0)
# gym.apply_dof_effort(env3, pole_dof_handle3, 200)
# ```
cart3_index = gym.find_actor_index(env3, "cartpole", gymapi.DOMAIN_ENV)

# find actor: cartpole2 dof: cart_to_pole index
# cart3_dof2_index = gym.find_actor_dof_index(env3, cartpole3, 'cart_to_pole', gymapi.DOMAIN_ENV)
# get total number of DoFs
num_dofs = gym.get_sim_dof_count(sim)
# create tensors for target efforts
cart3_effort_tensor = torch.zeros(cart_num_dofs, dtype=torch.float32, device="cpu")
# set effort for the pole
cart3_effort_tensor[1] = 200.0
cart3_target_position_tensor = torch.zeros(
    cart_num_dofs, dtype=torch.float32, device="cpu"
)
cart3_target_position_tensor[0] = 0.0
gym.set_dof_actuation_force_tensor_indexed(
    sim,
    gymtorch.unwrap_tensor(cart3_effort_tensor),
    gymtorch.unwrap_tensor(torch.tensor([cart3_index]).to(torch.int32)),
    len([cart3_index]),
)
gym.set_dof_position_target_tensor_indiced(
    sim,
    torch.unwrap_tensor(cart3_target_position_tensor),
    torch.unwrap_tensor(torch.tensor([cart3_index]).to(torch.int32)),
    len([cart3_index]),
)


# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

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

    # Update env 2: reverse cart target velocity when bounds reached
    # pos = gym.get_dof_position(env2, cart_dof_handle2)
    pos_tensor = gymtorch.unwrap_tensor(gym.acquire_dof_state_tensor(sim))
    pos = pos_tensor[cart2_index * cart_num_dofs, 0]

    if pos >= 0.5:
        # gym.set_dof_target_velocity(env2, cart_dof_handle2, -1.0)
        gym.set_dof_target_velocity_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(torch.tensor([-1.0])),
            gymtorch.unwrap_tensor(torch.tensor([cart2_index]).to(torch.int32)),
            len([cart2_index]),
        )
    elif pos <= -0.5:
        gym.set_dof_target_velocity(env2, cart_dof_handle2, 1.0)
        gym.set_dof_target_velocity_tensor_indexed(
            sim,
            gymtorch.unwrap_tensor(torch.tensor([-1.0])),
            gymtorch.unwrap_tensor(torch.tensor([cart2_index]).to(torch.int32)),
            len([cart2_index]),
        )

    # Update env 3: apply an effort to the pole to keep it upright
    pos = gym.get_dof_position(env3, pole_dof_handle3)
    gym.apply_dof_effort(env3, pole_dof_handle3, -pos * 50)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
