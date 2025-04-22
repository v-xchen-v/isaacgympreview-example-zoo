import numpy as np
from isaacgym import gymapi, gymutil

# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments and viewer options
args = gymutil.parse_arguments()
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.use_gpu = True

# Create sim
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# Create one env
num_envs = 1
spacing = 1.0
lower = gymapi.Vec3(-spacing, -spacing, 0.0)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 1)

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# (Optional) Add actor or asset here...

# Draw bounding box
min_box = np.array([0.4, -0.1, 0.4])
max_box = np.array([0.6,  0.1, 0.6])

from itertools import product, combinations
def draw_bounding_box(min_pt, max_pt, sim, viewer, color=(0, 1, 0)):
    # Convert color to gymapi.Vec3
    color_vec = gymapi.Vec3(color[0], color[1], color[2])
    
    corners_np = [np.array(corner) for corner in product(*zip(min_pt, max_pt))]
    corners = [gymapi.Vec3(c[0], c[1], c[2]) for c in corners_np]
    for i, j in combinations(range(8), 2):
        if np.sum(np.abs(corners_np[i] - corners_np[j]) > 1e-6) == 1:
            gymutil.draw_line(corners[i], corners[j], color=color_vec, gym=gym, env=env, viewer=viewer)

# Main loop
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    draw_bounding_box(min_box, max_box, sim=sim, viewer=viewer)

    gym.sync_frame_time(sim)

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
