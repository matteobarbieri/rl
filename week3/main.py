import gym
# import numpy as np
# from collections import namedtuple
# import collections
# import time

import argparse

from torch.utils.tensorboard import SummaryWriter

import atari_wrappers
from agent import DQNAgent

from datetime import datetime

# import utils

DQN_HYPERPARAMS = {
    'dueling': False,
    'noisy_net': False,
    'double_DQN': False,
    'n_multi_step': 2,
    'buffer_start_size': 10001,
    'buffer_capacity': 15000,
    'epsilon_start': 1.0,
    'epsilon_decay': 10**5,
    'epsilon_final': 0.02,
    'learning_rate': 5e-5,
    'gamma': 0.99,
    'n_iter_update_target': 1000
}


BATCH_SIZE = 32
MAX_N_GAMES = 3000
TEST_FREQUENCY = 10

# ENV_NAME = "PongNoFrameskip-v4"
SAVE_VIDEO = True
# DEVICE = 'cpu' #  or 'cuda'
DEVICE = 'cuda'  # or 'cuda'
SUMMARY_WRITER = True

LOG_DIR = 'tb_logs/runs'
# name = '_'.join([str(k)+'.'+str(v) for k, v in DQN_HYPERPARAMS.items()])
# name = 'prv'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("env_name", choices=[
        "PongNoFrameskip-v4",
        "SpaceInvaders-v4",
        "BreakoutNoFrameskip-v4"])

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # create the environment
    # env = atari_wrappers.make_env(ENV_NAME)
    env = atari_wrappers.make_env(args.env_name)

    # Create run name with environment name and timestamp of launch
    run_name = args.env_name+"_run_"+datetime.now().strftime("%Y%m%d_%H%M")

    if SAVE_VIDEO:
        # save the video of the games
        # env = gym.wrappers.Monitor(env, "main-"+args.env_name, force=True)
        # Save every 50th episode
        env = gym.wrappers.Monitor(
            env, "videos/"+args.env_name+"/run_"+datetime.now().strftime("%Y%m%d_%H%M"),  # noqa
            video_callable=lambda episode_id: episode_id % 50 == 0)

    # TensorBoard
    writer = SummaryWriter(log_dir=LOG_DIR+'/'+run_name) \
        if SUMMARY_WRITER else None

    print('Hyperparams:', DQN_HYPERPARAMS)

    # create the agent
    agent = DQNAgent(
        env, DQN_HYPERPARAMS, DEVICE, summary_writer=writer)

    n_games = 0
    # n_iter = 0

    # Play MAX_N_GAMES games
    while n_games < MAX_N_GAMES:

        obs = env.reset()
        done = False

        while not done:

            # act greedly
            action = agent.act_eps_greedy(obs)

            # one step on the environment
            new_obs, reward, done, _ = env.step(action)

            # add the environment feedback to the agent
            agent.add_env_feedback(obs, action, new_obs, reward, done)

            # sample and optimize NB: the agent could wait to have enough
            # memories
            agent.sample_and_optimize(BATCH_SIZE)

            obs = new_obs

        n_games += 1

        # print info about the agent and reset the stats
        agent.print_info()
        agent.reset_stats()

        # if n_games % TEST_FREQUENCY == 0:
            # print('Test mean:', utils.test_game(env, agent, 1))

    writer.close()


if __name__ == '__main__':
    main()
