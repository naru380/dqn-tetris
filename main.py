from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from agent import Agent
from utils import preprocess
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

RENDER = True
SIZE_RESIZED_IMAGE = 84
NUM_EPISODES = 10000
BATCH_SIZE = 128
GAMMA = 0.999
LEARNING_RATE = 0.001
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 10#00000
INTERVAL_UPDATE_POLICY_NET = 1
INTERVAL_UPDATE_TARGET_NET = 10000
INTERVAL_SAVE_MODEL = 1000000
REPLAY_MEMORY_SIZE = 400#000
PATH_LOGS_DIR = './logs'
PATH_MODELS_DIR = './models'
MY_SIMPLE_MOVEMENT = [
    ['A'],
    ['B'],
    ['right'],
    ['left'],
    ['down']
]



if __name__ == '__main__':

    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, MY_SIMPLE_MOVEMENT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        'device': device,
        'num_actions': env.action_space.n,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'gamma': GAMMA,
        'eps_start': EPS_START,
        'eps_end': EPS_END,
        'eps_decay': EPS_DECAY,
        'interval_update_policy_net': INTERVAL_UPDATE_POLICY_NET,
        'interval_update_target_net': INTERVAL_UPDATE_TARGET_NET,
        'interval_save_model': INTERVAL_SAVE_MODEL,
        'replay_memory_size': REPLAY_MEMORY_SIZE,
        'path_logs_dir': PATH_LOGS_DIR,
        'path_models_dir': PATH_MODELS_DIR
    }
    agent = Agent(params)

    if not os.path.isdir(PATH_LOGS_DIR):
        os.mkdir(PATH_LOGS_DIR)
    writer = SummaryWriter(PATH_LOGS_DIR + '/test')

    for episode in range(1, NUM_EPISODES+1):
        observation = env.reset()
        state = preprocess(observation, SIZE_RESIZED_IMAGE)
        t = done = total_rewards = total_loss = total_max_q_val = 0

        while True:

            if RENDER:
                env.render()

            action = agent.get_action(state)
            observation, reward, done, _ = env.step(action)

            if done:
                next_state = None
            else:
                next_state = preprocess(observation, SIZE_RESIZED_IMAGE)

            agent.memorize(state, action, next_state, reward)
            agent.train()

            state = next_state
            total_rewards += reward
            total_loss += agent.brain.loss
            total_max_q_val += float(agent.brain.q_vals.max(1)[0])

            if done or t > 100:
                writer.add_scalar("total_rewards", total_rewards, episode)
                writer.add_scalar("steps", total_rewards, episode)
                writer.add_scalar("avg_loss", total_loss/t, episode)
                writer.add_scalar("avg_max_q_val", total_max_q_val/t, episode)
                print('TIME_STEPS:{}, EPISODE:{}, T:{}, TOTAL_REWARDS:{}, AVG_LOSS:{}, AVG_MAX_Q_VAL:{}' \
                    .format(agent.brain.steps_done, episode, t, total_rewards, total_loss/t, total_max_q_val/t))
                t = done = total_rewards = total_loss = total_max_q_val = 0
                break

            t += 1
