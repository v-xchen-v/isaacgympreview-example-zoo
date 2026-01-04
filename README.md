# isaacgympreview-example-zoo
Provide standalone examples demonstrating IsaacGym features without requiring IsaacGymEnvs. Focuses on basic physics, sensors, and simple ready to go RL setups. Extended IsaacGymPreview python examples.

## Repository Structure
```
isaacgympreview-example-zoo/
│── examples/                 # Core IsaacGym preview examples
│   ├── physics_tests/        # Rigid body, collisions, friction, forces
│   ├── camera_sensors/       # Depth, RGB, segmentation, LiDAR examples
│   ├── soft_body/            # Cloth, rope, deformables
│   ├── terrain_navigation/   # Basic locomotion with terrains
│   ├── kinematics/           # Forward/inverse kinematics examples
│   ├── simple_manipulation/  # Pick-and-place, grasping basics
│── utils/                    # Common helper scripts
│   ├── visualization.py      # Plotting and rendering tools
│   ├── physics_utils.py      # Utility functions for forces, torque, collisions
│── docs/                     # Documentation & tutorials
│   ├── installation.md       # How to install IsaacGym and dependencies
│   ├── getting_started.md    # How to run example
│── requirements.txt          # Dependencies
│── README.md                 # Overview of repository
│── LICENSE                   # Open-source license

```

## Key Features
- No Dependency on IsaacGymEnvs -> Focus on raw IsaacGym Preview Functionality
- Beginner-friendly -> Small, focused examples with easy-to-follow code.
- Standalone examples -> Uses can copy a script and run it instantly.
- For learning & prototyping -> Good for understanding IsaacGym physics.

### Examples: Simple Force Application (Standalone)
Mimimal IsaacGym example, demonstrating force application.
```
import torch
from isaacgym import gymapi, gymutil, gymtorch

gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX)

# Create an environment and load an asset
env = gym.create_env(sim, gymapi.Vec3(-1, 0, 0), gymapi.Vec3(1, 2, 1), 1)
asset_options = gymapi.AssetOptions()
asset = gym.load_asset(sim, "urdf", "urdf/franka_panda.urdf", asset_options)
actor_handle = gym.create_actor(env, asset, gymapi.Transform(), "robot", 0, 1)

# Apply force
force = torch.tensor([10.0, 0.0, 0.0])
gym.apply_rigid_body_force_at_pos(sim, actor_handle, force.numpy(), [0, 0, 0])

# Simulate
for _ in range(100):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
```