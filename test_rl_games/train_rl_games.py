import rl_games
from rl_games.common import env_configurations
from rl_games.torch_runner import Runner
import gym
import custom_env  # Import to ensure env registration
import yaml
# from custom_env import CustomEnv

# env_configurations.register("my_task", {
#     "env_creator": lambda **kwargs: CustomEnv(**kwargs),
# })
# from rl_games.common.env_configurations import list_all_envs
print("Registered environments:", env_configurations.configurations.keys())


# def register_custom_env():
#     env_configurations.register(
#         "custom_env",
#         lambda config: gym.make(config["env_name"])
#     )

def load_config(yaml_file):
    """Load configuration from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # register_custom_env()
    
    runner = Runner()
    runner.load(load_config("test_rl_games/custom_env.yaml"))
    runner.run({
        'train': True
    })
