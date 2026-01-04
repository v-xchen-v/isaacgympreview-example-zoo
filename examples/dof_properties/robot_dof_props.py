import math
from isaacgym import gymapi
from isaacgym import gymutil

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

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

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
asset_file = "urdf/franka_description/robots/franka_panda.urdf"

# TODO: ?Seems buggy actuator API here, sometimes the get_asset_actuator_count got 0 by mistake. Ref: https://forums.developer.nvidia.com/t/gym-get-asset-actuator-count-does-not-return-the-actuator-count-from-urdf-assets/175691/4?
# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Find actuated dof names
num_robot_dofs = gym.get_asset_dof_count(robot_asset)
num_robot_actuated_dofs = gym.get_asset_actuator_count(robot_asset)

    
# load joint range information
robot_dof_props = gym.get_asset_dof_properties(robot_asset)
robot_dof_lower_limits = []
robot_dof_upper_limits = []
for i in range(num_robot_dofs):
    robot_dof_lower_limits.append(robot_dof_props["lower"][i])
    robot_dof_upper_limits.append(robot_dof_props["upper"][i])
print('Robot DOF lower limits:', robot_dof_lower_limits)
print('Robot DOF upper limits:', robot_dof_upper_limits)