from gym.envs.registration import register

register(
    id='math-v0',
    entry_point='gym_math_problems.envs:MathEnv',
)
