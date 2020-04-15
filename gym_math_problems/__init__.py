from gym.envs.registration import register

register(
    id='easy-sums-v0',
    entry_point='gym_math_problems.envs:EasySumsEnv',
)
register(
    id='hard-sums-v0',
    entry_point='gym_math_problems.envs:HardSumsEnv',
)
