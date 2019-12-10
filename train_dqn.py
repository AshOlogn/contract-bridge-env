import gym
import torch 
import torch.nn as nn
from torch import optim 
import numpy as np 
import numpy.ma as ma 

from contract_bridge.envs.bridge_trick_taking import BridgeEnv
from nfsp.nn import DQN, PG
from nfsp.buffers import ReplayMemory, Transition
from agents.agent import DQNAgent

import click
import random
import os

def train(n_episodes, num, epsilon=0.1):
    env = gym.make('contract_bridge:contract-bridge-v0')

    batch_size = 128
    gamma = 0.999
    target_update = 10000 # update target network after 1000 episodes
    steps_done = 0

    #DQN is p_00, p_01 is a teammate, the rest are opponents
    players = {}
    players['p_00'] = DQNAgent('p_00', env, epsilon=0.1, buffer_size=2000)
    players['p_01'] = DQNAgent('p_01', env, epsilon=0.1, buffer_size=2000)
    players['p_10'] = DQNAgent('p_10', env, epsilon=0.1, buffer_size=2000)
    players['p_11'] = DQNAgent('p_11', env, epsilon=0.1, buffer_size=2000)

    #to determine ordering
    order = ['p_00', 'p_11', 'p_01', 'p_10']

    sliding_window = []

    for episode in range(n_episodes):

        #reset the environment with a new random bid
        bid_level = random.randint(7,13)
        bid_trump = random.choice(['C', 'D', 'H', 'S', None])
        bid_trump = None
        bid_team = random.choice([0,1])
        env.reset(bid_level, bid_trump, bid_team)

        for r in range(13):

            for i in range(4):
                pid = env.current_player
                action = players[pid].act()
                env.play({'player': pid, 'card': action})
            
            (_, op_reward, _, _)  = env.step('p_10')
            (_, op_reward, _, _)  = env.step('p_11')
            (_, reward, _, _) = env.step('p_01')
            (_, reward, _, _) = env.step('p_00')

            if len(sliding_window) == 100:
                sliding_window.pop(0)
                sliding_window.append(1 if reward > op_reward else 0)
            else:
                sliding_window.append(1 if reward > op_reward else 0)
            
            if r == 12 and episode % 1000 == 0 and len(sliding_window) == 100:
                with open('logs/sliding-dqn/%d.txt' % num, 'a+') as f:
                    f.write('%d %f\n' % (episode, sum(sliding_window)/len(sliding_window)))
            

            #now update the networks of each of the agents
            players['p_00'].process_step(reward)
            players['p_01'].process_step(reward)
            players['p_10'].process_step(reward)
            players['p_11'].process_step(reward)

        #print("Episode {} completed".format(episode))
        # update the target network based on the current policy
        # also save the current policy network
        if episode % target_update == 0:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            torch.save(policy_dqn.state_dict(), "models/dqn/policy-network-%d.pth" % (episode))

    env.close()

if __name__ == '__main__':

    #make sure directories exist to save models and performance logs
    if not os.path.exists("models"):
        os.mkdir("models")
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
    
    if not os.path.exists("logs/sliding-dqn"):
        os.mkdir("logs")

    for i in range(1):
        print('starting %d...' % i)
        train(1000000, i)
    
    print('done')
