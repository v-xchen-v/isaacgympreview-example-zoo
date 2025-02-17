from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np
import torch

# Initialize gym and create simulation
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0, -9.8, 0)
sim_params.use_gpu_pipeline = False
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create an environment
env = gym.create_env(sim, gymapi.Vec3(-5, -5, -5), gymapi.Vec3(5, 5, 5), 1)

# Add a ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 1, 0)
gym.add_ground(sim, plane_params)

# Create a sphere asset
radius = 0.1  # Sphere radius
sphere_asset_options = gymapi.AssetOptions()
sphere_asset = gym.create_sphere(sim, radius, sphere_asset_options)

# Add the sphere actor to the environment
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0, 0.5, 0)  # Initial position of the sphere
sphere_actor = gym.create_actor(env, sphere_asset, initial_pose, "Sphere", 0, 0)

# Get the rigid body handle for the sphere
rigid_body_handle = gym.get_actor_rigid_body_handle(env, sphere_actor, 0)

# Apply a force to "throw" the sphere
_force = gymapi.Vec3(50, 500, 0)  # Force vector in Newtons
# force_position = gymapi.Vec3(0, 0.5, 0)  # Position where the force is applied
num_envs = 1
num_bodies = 1
device = "cpu"
forces = torch.tensor([[[50, 500, 0]]], device=device, dtype=torch.float)
# forces = torch.zeros((num_envs, num_bodies, 3), device=device, dtype=torch.float)

# viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, 3, 0), gymapi.Vec3(0, 0, 0))

# TODO: MUST call prepare_sim, otherwise the initial_root_states return all zeros, and use it to set the root, will got all nan states
gym.prepare_sim(sim)

# initial_body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_ALL)
# initial_body_states = gym.get_env_rigid_body_states(env, gymapi.STATE_ALL)
# initial_body_states = gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL) # [num_rigid_bodies, 3+4+3+3], position(3)+quat(4)+linear_vel(3)+angular_vel(4)
# TODO: MUST call refresh_actor_root_state_tensor, synchronize the state of the root bodies between the simulator and your code.
gym.refresh_actor_root_state_tensor(sim)
_initial_root_states = gym.acquire_actor_root_state_tensor(
    sim
)  # [num_rigid_bodies, 13]
initial_root_states = gymtorch.wrap_tensor(
    _initial_root_states
).clone()  # TODO: !MUST call clone() to store the intial data, the _initial_body_states is a handle to data and will refresh after step.
pass

step = 0
# Simulate and apply force
while True:
    if step == 1:  # Apply force at the first simulation step
        # Ref: https://docs.robotsfan.com/isaacgym/api/python/gym_py.html?highlight=apply_body_force_at_pos#isaacgym.gymapi.Gym.apply_body_force_at_pos
        # gym.apply_body_force_at_pos(env, rigid_body_handle, force, force_position, gymapi.LOCAL_SPACE)

        # Ref: https://docs.robotsfan.com/isaacgym/api/python/gym_py.html?highlight=apply_body_forces#isaacgym.gymapi.Gym.apply_body_forces
        # gym.apply_body_forces(env, rigid_body_handle, force, None, gymapi.LOCAL_SPACE)

        gym.apply_rigid_body_force_tensors(
            sim, gymtorch.unwrap_tensor(forces), None, gymapi.LOCAL_SPACE
        )

    def reset():
        # Reset the sphere to its initial position
        # gym.set_actor_rigid_body_states(env, actor_handle, initial_body_states, gymapi.STATE_ALL)
        # gym.set_env_rigid_body_states(env, initial_body_states, gymapi.STATE_ALL)
        print("Before Reset:")
        gym.refresh_actor_root_state_tensor(sim)
        # rigid not have DoF, use root state tensor
        print(gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)))

        gym.set_actor_root_state_tensor(
            sim, gymtorch.unwrap_tensor(initial_root_states)
        )

    if step % 100 == 0:
        print(f"Resetting at step {step}")
        reset()
        step = 0

    # Step simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)

    print("After Reset:")
    gym.refresh_actor_root_state_tensor(sim)
    print(gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim)))

    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

    step += 1

# Cleanup
gym.destroy_sim(sim)
