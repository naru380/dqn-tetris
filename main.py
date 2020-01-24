from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
# from preprocessing import Prepocessing
from agent import Agent

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

NUM_EPISODES = 10000
SIZE_RESIZED_IMAGE = 84

def preprocess(observation):
    state = Image.fromarray(np.uint8(observation))
    state = state.resize((SIZE_RESIZED_IMAGE, SIZE_RESIZED_IMAGE))
    state = state.convert('L')
    state = np.asarray(state)
    state = state / 255.0
    state = state[np.newaxis, :, :]
    return state



env = gym_tetris.make('TetrisA-v0')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
MY_SIMPLE_MOVEMENT = [
    ['A'],
    ['B'],
    ['right'],
    ['left'],
    ['down']
]
env = JoypadSpace(env, MY_SIMPLE_MOVEMENT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(env, device)

#prep = Prepocessing()
for episode in range(NUM_EPISODES):
    observation = env.reset()
    state = preprocess(observation)

    t = 0
    done = False
    total_reward = 0

    while True:
        t += 1

        action = int(agent.select_action(state).cpu().numpy())
        observation, reward, done, _ = env.step(action)

        env.render()

        if done:
            next_state = None
        else:
            next_state = preprocess(observation)

        agent.memorize(state, action, next_state, reward)

        agent.optimize()

        state = next_state
        total_reward += reward

        if done:
            print('Episode:{}, Time_Steps:{}, Total_Reward:{}'.format(episode, t, total_reward))
            t = 0
            done = False
            total_reward = 0
            break
