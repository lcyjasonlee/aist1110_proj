from gym.envs.registration import register

register(
    id="the_floor_is_lava-v1",
    entry_point="the_floor_is_lava.envs:MainEnv",
)
