from nes_py.wrappers import JoypadSpace
from network import QNetwork
from utils import preprocess
import gym_tetris
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

    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, MY_SIMPLE_MOVEMENT)

    params = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_actions': env.action_space.n,
        'path_model': path_model,
    }

    agent = TrainedAgent(params)

    done = True
    for step in range(5000):
        if done:
            observation = env.reset()
        state = preprocess(observation, SIZE_RESIZED_IMAGE)
        observation, reward, done, _ = env.step(agent.get_action(state))
        env.render()

    env.close()
