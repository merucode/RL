from gym.envs.registration import register

register(
    id='TraderWorld-v0',
    entry_point='trader_world.envs:TraderWorldEnv',
)