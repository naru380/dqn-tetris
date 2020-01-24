from network import QNetwork
from replay_memory import ReplayMemory, Transition
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import math

BATCH_SIZE = 128
GAMMA = 0.999
LEARNING_RATE = 0.001
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 200
POLICY_NETWORK_UPDATE_INTERVAL = 1
TARGET_NETWORK_UPDATE_INTERVAL = 10000
REPLAY_MEMORY_SIZE = 400000



class Agent():

    def __init__(self, env, device):
        self.num_action = env.action_space.n
        self.device = device
        self.policy_net = QNetwork(self.num_action).to(self.device)
        self.target_net = QNetwork(self.num_action).to(self.device)
        self.steps_done = 0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(REPLAY_MEMORY_SIZE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)


    def select_action(self, state):
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.from_numpy(state).float().to(self.device).unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_action)]], device=self.device, dtype=torch.long)

    def optimize(self):
        if (len(self.memory) < BATCH_SIZE) or (self.steps_done % POLICY_NETWORK_UPDATE_INTERVAL != 0):
            pass
        else:
            transitions = self.memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in torch.tensor(batch.next_state, device=self.device, dtype=torch.float) if s is not None])

            state_batch = torch.cat([torch.tensor(batch.state, device=self.device, dtype=torch.float)])
            action_batch = torch.cat([torch.tensor(batch.action, device=self.device, dtype=torch.long)])
            reward_batch = torch.cat([torch.tensor(batch.reward, device=self.device, dtype=torch.int)])

            state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

            next_state_values = torch.zeros(BATCH_SIZE, device=self.device)

            next_state_values[non_final_mask] = self.target_net(non_final_next_states.unsqueeze(1)).max(1)[0].detach()

            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if self.steps_done % TARGET_NETWORK_UPDATE_INTERVAL == 0:
                self.update_target_network()

        self.steps_done += 1

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
