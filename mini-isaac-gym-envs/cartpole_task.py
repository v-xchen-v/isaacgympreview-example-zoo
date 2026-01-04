from .vec_task import MyVecTask

from rl_games.common import env_configurations
env_configurations.register("MyVecTask", {
    "env_creator": lambda **kwargs: MyVecTask(**kwargs),
})