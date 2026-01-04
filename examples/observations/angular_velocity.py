import isaacgym
import isaacgym.gymapi as gymapi
import isaacgym.gymutil as gymutil
import numpy as np
import time

# Initialize Gym
gym = gymapi.acquire_gym()

# Simulation parameters
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0, -9.8, 0)
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # Gravity

# Choose a physics engine
# sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1

# Create simulator
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create a spherical asset
asset_root = './assets'
object_asset_file = "urdf/textured_sphere/sphere.urdf"
asset_options = gymapi.AssetOptions()
asset_options.disable_gravity = False  # Enable gravity
asset_options.use_mesh_materials = True
# sphere_asset = gym.create_sphere(sim, 0.2, asset_options)  # 0.2m radius
sphere_asset = gym.load_asset(sim, asset_root, object_asset_file)
if sphere_asset is None:
    raise Exception("Failed to load object asset")

# Create an environment
env = gym.create_env(sim, gymapi.Vec3(-5, -5, -5), gymapi.Vec3(5, 5, 5), 1)

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Sphere initial pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.2)  # Slightly above ground
pose.r = gymapi.Quat(0, 0, 0, 1)

# Add sphere actor to environment
sphere_handle = gym.create_actor(env, sphere_asset, pose, "sphere", 0, 1)
scale = 2.0
gym.set_actor_scale(env, sphere_handle, scale)

# Get actor index
actor_idx = gym.get_actor_index(env, sphere_handle, gymapi.DOMAIN_SIM)

# Simulation loop
gym.prepare_sim(sim)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, 3, 0), gymapi.Vec3(0, 0, 0))
for i in range(2000):  # Run for 200 steps
    # Apply torque to spin the sphere
    # torque = np.array([0.0, 0.0, 0.5], dtype=np.float32)  # Rotate around Z-axis
    torque = gymapi.Vec3(0, 0., 0.05)  # Force vector in Newtons
    gym.apply_body_forces(env, sphere_handle, None, torque, gymapi.LOCAL_SPACE)

    # Step simulation
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)

    # Get angular velocity
    rb_state = gym.get_actor_rigid_body_states(env, sphere_handle, gymapi.STATE_ALL)
    angular_velocity = rb_state['vel']['angular']  # Extract angular velocity (w_x, w_y, w_z)

    print(f"Step {i}, Angular Velocity: {angular_velocity}")
    
    ang_vel = np.array([float(x) for x in angular_velocity[0]])
    ang_vel_norm = np.linalg.norm(ang_vel)
    ang_vel_norm_value = np.exp(-2.0 * ang_vel_norm)
    # measure how close the ang_vel_norm close to a constant value
    
    
    # shpere_transform = gymapi.Transform()
    # shpere_transform.p = gymapi.Vec3(*rb_state['pose']['p'][0])
    # shpere_transform.r = gymapi.Quat(*rb_state['pose']['r'][0])
    # sphere_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, shpere_transform, color=(1, 1, 0))
    # gymutil.draw_lines(sphere_geom, gym, viewer, env, shpere_transform)


    # Delay for better visualization
    time.sleep(0.02)

    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    
# Cleanup
gym.destroy_sim(sim)
