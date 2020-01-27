from nes_py.wrappers import JoypadSpace
from network import QNetwork
from utils import preprocess
from gym import wrappers
import gym_tetris
import os
import sys
import torch

MY_SIMPLE_MOVEMENT = [
    ['A'],
    ['B'],
    ['right'],
    ['left'],
    ['down']
]

SIZE_RESIZED_IMAGE = 84

NUM_EPISODES = 1

PATH_MOVIES_DIR = './movies'


class TrainedBrain():

    def __init__(self, parmas):
        self.num_actions = params['num_actions']
        self.device = params['device']
        self.path_model = params['path_model']
        self.policy_net = QNetwork(self.num_actions).to(self.device)
        self.policy_net.load_state_dict(torch.load(self.path_model, map_location=self.device))
        self.policy_net.eval()


    def decide_action(self, state):
        with torch.no_grad():
            self.q_vals = self.policy_net(torch.from_numpy(state.copy()).float().to(self.device).unsqueeze(0))

        return int(self.q_vals.max(1)[1].view(1, 1))



class TrainedAgent():

    def __init__(self, params):
        self.brain = TrainedBrain(params)


    def get_action(self, state):
        return self.brain.decide_action(state)



if __name__ == '__main__':
    args = sys.argv
    path_model = args[1]
    save_interval = int(args[2])

    if not os.path.isdir(PATH_MOVIES_DIR):
        os.makedirs(PATH_MOVIES_DIR)

    env = gym_tetris.make('TetrisA-v0')
    env = wrappers.Monitor(env, PATH_MOVIES_DIR, video_callable=lambda ep: ep%save_interval==0, force=True)
    env = JoypadSpace(env, MY_SIMPLE_MOVEMENT)

    params = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_actions': env.action_space.n,
        'path_model': path_model,
    }

    agent = TrainedAgent(params)

    for i in range(NUM_EPISODES):
        done = False
        observation = env.reset()
        while True:
            env.render()
            state = preprocess(observation, SIZE_RESIZED_IMAGE)
            observation, reward, done, _ = env.step(agent.get_action(state))
            if done:
                break

    env.close()
