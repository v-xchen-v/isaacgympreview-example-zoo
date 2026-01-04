'''Example of listing DOF names of a robot(an asset or an actor)'''

import isaacgym
from isaacgym import gymapi

# Initialize Gym
gym = gymapi.acquire_gym()

# Create a simulation environment
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Load an asset (robot model)
asset_root = "./assets"
asset_file = "urdf/cartpole.urdf"  # Change to your robot file

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True

asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Get DOF names
dof_names = gym.get_asset_dof_names(asset)

# Print DOF names
print("List of DOF names:", dof_names)

# List index and corresponding DOF name
for i, name in enumerate(dof_names):
    print(f"Index {i}: {name}")


# Create an actor
initial_pose = gymapi.Transform()
# Create environment 1
env = gym.create_env(sim, gymapi.Vec3(-1, -1, 0), gymapi.Vec3(1, 1, 1), 1)
actor_handle = gym.create_actor(env, asset, initial_pose, "actor", 0, 1)

# Get DOF names of the actor
actor_dof_names = gym.get_actor_dof_names(env, actor_handle)

# Print DOF names of the actor
print("List of actor DOF names:", actor_dof_names)