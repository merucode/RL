from gym.envs.registration import register

register(
    id='TradeWorld-v0',
    entry_point='gym_examples.envs:TradeWorldEnv',
)