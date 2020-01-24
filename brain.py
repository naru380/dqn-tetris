from network import QNetwork
from replay_memory import ReplayMemory, Transition
import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX  import SummaryWriter
import math
import random



class Brain:

    def __init__(self, params):
        self.num_actions = params['num_actions']
        self.device = params['device']
        self.batch_size = params['batch_size']
        self.learning_rate = params['learning_rate']
        self.gamma = params['gamma']
        self.eps_start = params['eps_start']
        self.eps_end = params['eps_end']
        self.eps_decay = params['eps_decay']
        self.policy_net = QNetwork(self.num_actions).to(self.device)
        self.target_net = QNetwork(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(params['replay_memory_size'])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.steps_done = 0


    def decide_action(self, state):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        sample = random.random()
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.from_numpy(state).float().to(self.device).unsqueeze(0)).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)


    def optimize(self):
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([torch.tensor(s, device=self.device, dtype=torch.float) for s in batch.next_state if s is not None])

        state_batch = torch.cat([torch.tensor(batch.state, device=self.device, dtype=torch.float)])
        action_batch = torch.cat([torch.tensor(batch.action, device=self.device, dtype=torch.long)])
        reward_batch = torch.cat([torch.tensor(batch.reward, device=self.device, dtype=torch.int)])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target_net(non_final_next_states.unsqueeze(1)).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
