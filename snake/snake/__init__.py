from gym.envs.registration import register

register(
    id='snake-v0',
    entry_point='snake.envs:SnakeEnv',
)
