from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from agent import Agent
from utils import preprocess
import numpy as np
import torch


SIZE_RESIZED_IMAGE = 84
NUM_EPISODES = 10000

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
    state = preprocess(observation, SIZE_RESIZED_IMAGE)

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
            next_state = preprocess(observation, SIZE_RESIZED_IMAGE)

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
