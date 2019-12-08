import gym
import torch 
import torch.nn as nn
from torch import optim 

from contract_bridge.envs.bridge_trick_taking import BridgeEnv
from nfsp.nn import DQN, PG
from nfsp.buffers import ReplayMemory, Transition
from agents.agent import SmartGreedyAgent

import click
import random
import os

@click.command()
@click.option("--n_episodes", type=int, default=10000)
def train(n_episodes, epsilon=0.1):
    env = gym.make('contract_bridge:contract-bridge-v0')

    batch_size = 32
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 1000 # update target network after 1000 episodes
    steps_done = 0

    device = torch.device('cpu')
    #check after
    policy_dqn = DQN().to(device)
    target_dqn = DQN().to(device)
    target_dqn.load_state_dict(policy_dqn.state_dict())

    optimizer = optim.RMSprop(policy_dqn.parameters())
    criterion = nn.SmoothL1Loss()
    replay_memory = ReplayMemory(130)

    #DQN is p_00, p_01 is a teammate, the rest are opponents
    players = {}
    players['p_01'] = SmartGreedyAgent('p_01', env)
    players['p_10'] = SmartGreedyAgent('p_10', env)
    players['p_11'] = SmartGreedyAgent('p_11', env)

    #to determine ordering
    order = ['p_00', 'p_11', 'p_01', 'p_10']

    for episode in range(n_episodes):

        #reset the environment with a new random bid
        bid_level = random.randint(7,13)
        bid_trump = random.choice(['C', 'D', 'H', 'S', None])
        bid_team = random.choice([0,1])
        env.reset(bid_level, bid_trump, bid_team)

        for r in range(13):
            prev_dqn_state = None
            dqn_action = None

            for i in range(4):
                #the environment tells us which player should go next
                pid = env.current_player

                if pid == 'p_00':
                    #dqn agent
                    prev_dqn_state = env.get_state('p_00').to(device)

                    #Epsilon-greedy action selection
                    if random.random() < 0:
                        #random action (exploration)

                        #convert this action to a tensor
                        dqn_action = random.choice(env.hands['p_00'])

                        env.play({'player': 'p_00', 'card': dqn_action})

                    else:
                        #get the dqn output
                        output = policy_dqn.forward(prev_dqn_state)

                        #mask out output[:mask_out_count] = (some high negative number)
                        pred = nn.functional.softmax(output)

                        #loop through hand and pick card with highest dqn output
                        hand = env.hands['p_00']

                        # dqn_action = hand[0]
                        # score = output[0]
                        score, dqn_action = torch.max(pred, dim = 0)
                        print(dqn_action.item())
                        #chosen out of index
                        print(hand)
                        card = env.hands['p_00'][dqn_action.item()]
                        # for card in hand[1:]:
                        #     if output[env.card_to_index[card]] > score:
                        #         dqn_action = card 
                        #         score = output[env.card_to_index[card]]
                        
                        env.play({'player': 'p_00', 'card': card})

                else:
                    card = players[pid].act()
                    env.play({'player': pid, 'card': card})
            
            env.step('p_01')
            env.step('p_10')
            env.step('p_11')

            (obs, reward, done, info) = env.step('p_00')

            if r== 12 and episode % 50 == 0:
                print(reward)

            env.current_trick = []
            reward_tensor = torch.tensor([reward], device=device)
            next_dqn_state = env.get_state('p_00')
            print(type(next_dqn_state), type(dqn_action), type(prev_dqn_state), type(reward_tensor)) 
            replay_memory.push(prev_dqn_state.unsqueeze(dim = 0), dqn_action.unsqueeze(dim = 0), reward_tensor.unsqueeze(dim = 0), next_dqn_state.unsqueeze(dim = 0))

            if len(replay_memory) > batch_size:
                #get transitions of size "batch_size"
                transitions = replay_memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                    device=device, dtype=torch.uint8)

                next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
                state_batch = torch.cat(batch.state)
                next_state_batch = torch.cat(batch.next_state)
                action_batch = torch.cat(batch.action)
                print(action_batch)
                print(action_batch.shape)
                reward_batch = torch.cat(batch.reward)
                
                # print(policy_dqn(state_batch))
                q_values = policy_dqn(state_batch.to(device))
                q_values = nn.functional.softmax(q_values, dim = 1)
                print(q_values.shape)

                q_values = q_values.gather(- 1, action_batch.unsqueeze(dim = 1))
                
                next_state_values = torch.zeros(batch_size, device=device)
                next_out = target_dqn(next_states)
                print(next_out.shape)
                next_state_values[mask] = next_out.max(1)[0].detach()
                expected_q_values = (next_state_values * gamma) + reward_batch.float()

                #compute loss
                loss = criterion(q_values, expected_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                for param in policy_dqn.parameters():
                    param.grad.data.clamp_(-1, 1)
                
                optimizer.step()
        
        #print("Episode {} completed".format(episode))
        # update the target network based on the current policy
        # also save the current policy network
        if episode % target_update == 0:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            torch.save(policy_dqn.state_dict(), "models/policy-network-{}.pth".format(episode))

    env.close()


if __name__ == '__main__':

    #make sure directories exist to save models and performance logs
    if not os.path.exists("models"):
        os.mkdir("models")
    
    if not os.path.exists("logs"):
        os.mkdir("logs")
    
    train()
