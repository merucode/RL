from gym.envs.registration import register

register(
    id='TradeWorld-v0',
    entry_point='gym_examples.envs:TradeWorldEnv',
    max_episode_steps=300,
)