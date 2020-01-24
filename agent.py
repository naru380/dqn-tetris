from brain import Brain
import os
import torch



class Agent():

    def __init__(self, params):
        self.brain = Brain(params)
        self.batch_size =  params['batch_size']
        self.interval_update_policy_net = params['interval_update_policy_net']
        self.interval_update_target_net = params['interval_update_target_net']
        self.interval_save_model = params['interval_save_model']
        self.path_models_dir = params['path_models_dir']


    def _update_policy_network(self):
        self.brain.optimize()


    def _update_target_network(self):
        self.brain.update_target_network()


    def get_action(self, state):
        return int(self.brain.decide_action(state).cpu().numpy())


    def memorize(self, state, action, next_state, reward):
        self.brain.memory.push(state, action, next_state, reward)


    def train(self):
        if (len(self.brain.memory) < self.batch_size) \
            or (self.brain.steps_done % self.interval_update_policy_net != 0):
            pass
        else:
            self._update_policy_network()

            if self.brain.steps_done % self.interval_update_target_net == 0:
                self._update_target_network()

        if self.brain.steps_done % self.interval_save_model == 0:
            if not os.path.isdir(self.path_models_dir):
                os.makedirs(self.path_models_dir)
            torch.save(self.brain.policy_net.state_dict(), self.path_models_dir + '/' + str(self.brain.steps_done))
            print("Saved Policy Network's Weights.")
