from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from agent import Agent
from utils import preprocess
import numpy as np
import torch

RENDER = True

SIZE_RESIZED_IMAGE = 84
NUM_EPISODES = 10000
BATCH_SIZE = 128
GAMMA = 0.999
LEARNING_RATE = 0.001
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000
INTERVAL_UPDATE_POLICY_NET = 1
INTERVAL_UPDATE_TARGET_NET = 10000
REPLAY_MEMORY_SIZE = 400000
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
        'replay_memory_size': REPLAY_MEMORY_SIZE
    }
    agent = Agent(params)

    for episode in range(NUM_EPISODES):
        observation = env.reset()
        state = preprocess(observation, SIZE_RESIZED_IMAGE)
        t = done = total_reward = 0

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
            total_reward += reward

            if done:
                print('Episode:{}, Time_Steps:{}, Total_Reward:{}'.format(episode, t, total_reward))
                t = done = total_reward = 0
                break
