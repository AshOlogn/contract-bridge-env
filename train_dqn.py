import gym
import torch 
import torch.nn as nn
from torch import optim 
import numpy as np 
import numpy.ma as ma

from contract_bridge.envs.bridge_trick_taking import BridgeEnv
from networks.nn import DQN
from networks.buffers import ReplayMemory, Transition
from agents.agent import DQNAgent

import random
import os


def train(n_episodes, epsilon=0.1):
    env = gym.make('contract_bridge:contract-bridge-v0')
    target_update = 1000 # update target network after 1000 episodes

    #DQN is p_00, p_01 is a teammate, the rest are opponents
    players = {}
    players['p_00'] = DQNAgent('p_00', env, epsilon=0.1, buffer_size=100000)
    players['p_01'] = DQNAgent('p_01', env, epsilon=0.1, buffer_size=100000)
    players['p_10'] = DQNAgent('p_10', env, epsilon=0.1, buffer_size=100000)
    players['p_11'] = DQNAgent('p_11', env, epsilon=0.1, buffer_size=100000)

    #to determine ordering
    order = ['p_00', 'p_11', 'p_01', 'p_10']

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
            
            #now update the networks of each of the agents
            players['p_00'].process_step(reward)
            players['p_01'].process_step(reward)
            players['p_10'].process_step(reward)
            players['p_11'].process_step(reward)

            if r == 12 and episode % 1000 == 0 and players['p_00'].last_loss is not None:
                for pid in order:
                    with open('logs/dqn/%s.txt' % pid, 'a+') as f:
                        f.write('%d %f\n' % (episode, players[pid].last_loss))
        
        # update the target network based on the current policy
        # also save the current policy network
        if episode % target_update == 0:
            for pid in order:
                players[pid].target_update()
        
        if episode % 10000 == 0:
            for pid in order:
                torch.save(players[pid].policy_dqn.state_dict(), "models/dqn/%s/policy-network-%d.pth" % (pid, episode))

    env.close()

if __name__ == '__main__':

    #make sure directories exist to save models and performance logs
    if not os.path.exists("models"):
        os.mkdir("models")
    
    if not os.path.exists("models/dqn"):
        os.mkdir("models/dqn")
    
    for pid in ['p_00', 'p_11', 'p_01', 'p_10']:
        if not os.path.exists("models/dqn/" + pid):
            os.mkdir("models/dqn/" + pid)
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
    
    if not os.path.exists("logs/dqn"):
        os.mkdir("logs/dqn")
    
    
    train(1000000)
    print('done')
