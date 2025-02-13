import isaacgym
from isaacgym import gymapi

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)


asset_root = "../../assets"
asset_options = gymapi.AssetOptions()
robot_asset = gym.load_asset(sim, asset_root, "urdf/cartpole.urdf", asset_options)