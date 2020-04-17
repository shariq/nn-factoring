from gym_math_problems.envs.math_env import MathEnv
import numpy as np

env = MathEnv(difficulty=0.5, input_size=4, output_size=4, arg_sizes=(2,2), base=10, verbose=2)
env.seed(0)

while True:
    observation = env.reset()
    while True:
        print('observation or concat(input[{}], output[{}])=\n'.format(env.input_size, env.output_size), observation)
        action = np.array([int(e) for e in str(input('action ({} integers max value {} with commas in between):\n'.format(env.input_size, env.base))).strip().split(',')])
        while action.shape[0] != env.input_size:
            print('bad action!!! action.shape[0]={}!=env.input_size={}'.format(action.shape[0], env.input_size))
            action = np.array(int(e) for e in str(input('action ({} integers max value {} with commas in between):\n'.format(env.input_size, env.base))).split(','))
        print('================')
        observation, reward, done, _ = env.step(action)
        print('reward=', reward)
        if done:
            print('done!\n\n\n')
            break