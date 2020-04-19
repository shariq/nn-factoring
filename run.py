import os

from gym import spaces
import numpy as np
from gym_math_problems.envs.math_env import MathEnv

from stable_baselines import PPO2
from stable_baselines import bench
from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.math_util import safe_mean

# size of LSTM
N_LSTM = 256
# architecture of network
# NET_ARCH = [512, 'lstm', 512]
NET_ARCH = ['lstm', 256]
# NET_ARCH=['lstm', 512, dict(vf=[256], pi=[256])]
# layer norm the LSTM
LAYER_NORM = False

N_TRIALS = 100
NUM_ENVS = 64
NUM_EPISODES_FOR_SCORE = 10

class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=N_LSTM, reuse=reuse, net_arch=NET_ARCH,
                         layer_norm=LAYER_NORM, feature_extraction="mlp", **_kwargs)

def make_env(i):
    env = MathEnv(difficulty=1.0, base=10, input_size=4, output_size=4, arg_sizes=(1,2), verbose=0)
    # env = bench.Monitor(env, None, allow_early_resets=True)
    env.seed(i)
    env = bench.Monitor(env, None, allow_early_resets=True)
    return env

if __name__ == '__main__':
    print('making envs')
    env = SubprocVecEnv([lambda: make_env(i) for i in range(NUM_ENVS)])
    env = VecNormalize(env)
    print('made envs')

    model = PPO2(CustomLSTMPolicy, env, n_steps=32000, nminibatches=NUM_ENVS, lam=0.95, gamma=0.999,
                    noptepochs=10, ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, verbose=1)
    print('initialized ppo2')

    # eprewmeans = []
    # def reward_callback(local, _):
    #     eprewmeans.append(safe_mean([ep_info['r'] for ep_info in local['ep_info_buf']]))

    print('running model.learn')
    # model.learn(total_timesteps=100000, callback=reward_callback)
    model.learn(total_timesteps=1000000)

    # average_reward = sum(eprewmeans[-NUM_EPISODES_FOR_SCORE:]) / NUM_EPISODES_FOR_SCORE
    # print('episode rewards for last {} episodes:\n{}'.format(NUM_EPISODES_FOR_SCORE, eprewmeans[-NUM_EPISODES_FOR_SCORE:]))
