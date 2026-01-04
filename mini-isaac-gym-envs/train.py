import yaml
import argparse
from vec_task import MyVecTask
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations
import torch

env_configurations.register("my_task", {
    "env_creator": lambda **kwargs: MyVecTask(**kwargs),
})
# from rl_games.common.env_configurations import list_all_envs
print("Registered environments:", env_configurations.configurations.keys())


def load_config(yaml_file):
    """Load configuration from a YAML file."""
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser(description="Train PPO using rl_games")
    parser.add_argument("--config", type=str, default="mini-isaac-gym-envs/cartpole_config.yaml", help="Path to the YAML configuration file")
    # parser.add_argument("--train", action="store_true", help="Run training mode")
    # parser.add_argument("--play", action="store_true", help="Run evaluation mode")
    
    args = parser.parse_args()
    
    # Load YAML configuration
    config = load_config(args.config)
    
    # Ensure torch device is set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)

    # Set up runner with observer
    runner = Runner()
    runner.load(config)
    
    # if args.train:
    runner.run({"train": True})
    # elif args.play:
    #     runner.run({"play": True})

if __name__ == "__main__":
    main()
