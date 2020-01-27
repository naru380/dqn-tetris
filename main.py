from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from agent import Agent
from utils import preprocess
import datetime
import os
import shutil
import json
import numpy as np
import torch
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':

    json_params = json.load(open('./params.json', 'r'))

    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, json_params['action_space'])

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    params = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'num_actions': env.action_space.n,
        'batch_size': json_params['batch_size'],
        'learning_rate': json_params['learning_rate'],
        'gamma': json_params['gamma'],
        'eps_start': json_params['eps_start'],
        'eps_end': json_params['eps_end'],
        'eps_decay': json_params['eps_decay'],
        'interval_update_policy_net': json_params['interval_update_policy_net'],
        'interval_update_target_net': json_params['interval_update_target_net'],
        'interval_save_model': json_params['interval_save_model'],
        'replay_memory_size': json_params['replay_memory_size'],
        'path_logs_dir': json_params['path_logs_root_dir']+'/'+now,
        'path_models_dir': json_params['path_models_root_dir']+'/'+now
    }
    agent = Agent(params)

    if not os.path.isdir(params['path_logs_dir']):
        os.makedirs(params['path_logs_dir'])
    shutil.copy('./params.json', params['path_logs_dir']+'/params.json')
    writer = SummaryWriter(params['path_logs_dir'])
    dummy_input_to_policy_net = torch.randn(1, json_params['size_resized_image'], json_params['size_resized_image']).float().to(params['device']).unsqueeze(0)
    dummy_input_to_target_net = torch.randn(1, json_params['size_resized_image'], json_params['size_resized_image']).float().to(params['device']).unsqueeze(0)
    writer.add_graph(agent.brain.policy_net, dummy_input_to_policy_net)
    writer.add_graph(agent.brain.target_net, dummy_input_to_target_net)

    for episode in range(1, json_params['num_episodes']+1):
        observation = env.reset()
        state = preprocess(observation, json_params['size_resized_image'])
        t = done = total_rewards = total_loss = total_max_q_val = 0

        while True:
            if json_params['render']:
                env.render()

            t += 1
            action = agent.get_action(state)
            observation, reward, done, _ = env.step(action)
            if done:
                next_state = None
            else:
                next_state = preprocess(observation, json_params['size_resized_image'])

            agent.memorize(state, action, next_state, reward)
            agent.train()

            state = next_state
            total_rewards += reward
            total_loss += agent.brain.loss
            total_max_q_val += float(agent.brain.q_vals.max(1)[0])

            if done:
                writer.add_scalar("total_rewards", total_rewards, episode)
                writer.add_scalar("steps", total_rewards, episode)
                writer.add_scalar("avg_loss", total_loss/t, episode)
                writer.add_scalar("avg_max_q_val", total_max_q_val/t, episode)
                print('TIME_STEPS:{}, EPISODE:{}, T:{}, TOTAL_REWARDS:{}, AVG_LOSS:{}, AVG_MAX_Q_VAL:{}' \
                    .format(agent.brain.steps_done, episode, t, total_rewards, total_loss/t, total_max_q_val/t))
                t = done = total_rewards = total_loss = total_max_q_val = 0
                break
