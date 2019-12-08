import gym
import torch 
import torch.nn as nn
from torch import optim 

from contract_bridge.envs.bridge_trick_taking import BridgeEnv
from nfsp.nn import DQN, PG
from nfsp.buffers import ReplayBuffer, ReservoirBuffer, Transition
from agents.agent import SmartGreedyAgent

import click
import random


def take_action(pid, env, prev_state, dqn_action, pgn_action, best):
    buffer_size = 10000
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10 # update target network after 10 episodes
    steps_done = 0
    epsilon = 0.1

    #revisit this line
    if not best:
        #E-greedy. Maybe change
        if random.random() < epsilon:
            #random action (exploration)
            model_action = random.choice(env.hands[pid])
            env.play({'player': 'p_00', 'card': model_action})
        else:
            #get the dqn output. Regover this logic maybe since it's a returned distribution.
            output = pgn_action.forward(env.get_state(pid))

            #loop through hand and pick card with highest output
            hand = env.hands[pid]
            card_to_index = env.card_to_index

            best_card = hand[0]
            score = output[0]
            for card in hand[1:]:
                if output[card_to_index[card]] > score:
                    best_card = card 
                    score = output[card_to_index[card]]

            model_action = best_card
            env.play({'player': pid, 'card': best_card})
    else:
        if random.random() < epsilon:
            #random action (exploration)
            model_action = random.choice(env.hands[pid])
            env.play({'player': 'p_00', 'card': model_action})
        else:
            #maybe
            eps = eps_final + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)
            output = dqn_action.forward(env.get_state(pid))

            #loop through hand and pick card with highest output
            hand = env.hands[pid]
            card_to_index = env.card_to_index

            best_card = hand[0]
            score = output[0]
            for card in hand[1:]:
                if output[card_to_index[card]] > score:
                    best_card = card 
                    score = output[card_to_index[card]]

            model_action = best_card
            env.play({'player': pid, 'card': best_card})

    return prev_state, model_action

def multi_step_reward(rewards, gamma):
    ret = 0.
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret

def rl_loss(policy, target, replay_buffer, p_optimizer):
    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))
    mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_dqn_state)), device=device, dtype=torch.uint8)

    next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    next_state_batch = torch.cat(batch.next_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    q_values = policy(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[mask] = target(next_states).max(1)[0].detach()
    expected_q_values = (next_state_values * gamma) + reward_batch

    #compute loss
    loss = criterion(q_values, expected_q_values.unsqueeze(1))

    p_optimizer.zero_grad()
    loss.backward()
    for param in policy_dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    p_optimizer.step()

    return loss

def sl_loss(policy, target, replay_buffer, p_optimizer):
    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))
    mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_dqn_state)), device=device, dtype=torch.uint8)

    next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    next_state_batch = torch.cat(batch.next_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    probs = policy(state_batch)
    probs_with_actions = probs.gather(1, action_batch.unsqueeze(1))
    log_probs = probs_with_actions.log()

    loss = -1 * log_probs.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

@click.command()
@click.option("--n_episodes", type=int, default=1)
def train(n_episodes, env, epsilon = 0.1):
    buffer_size = 10000
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10 # update target network after 10 episodes
    steps_done = 0
    eta = 0.1

    memory_size = 1024  # for replay-buffer
    batch_size = 64  # for mini-batch

    # === configuration for training ===
    iteration = 512

    # === configuration for epsilon-greedy ===
    eps_decay = 0.002
    eps_high = 1.0

    # === configuration for learning rate decay ===
    learning_rate = 1e-4
    learning_decay = 1e-5

    # === configuration for saving ===
    max_to_keep = 10
    train_freq = 1
    rl_start = 10000
    sl_start = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #create policy/target networks (RL) for each of the 4 agents
    p00_policy_model = DQN().to(device)
    p00_target_model = DQN().to(device)
    p00_target_model.load_state_dict(p00_policy_model.state_dict())

    p01_policy_model = DQN().to(device)
    p01_target_model = DQN().to(device)
    p01_target_model.load_state_dict(p01_policy_model.state_dict())

    p10_policy_model = DQN().to(device)
    p10_target_model = DQN().to(device)
    p10_target_model.load_state_dict(p10_policy_model.state_dict())

    p11_policy_model = DQN().to(device)
    p11_target_model = DQN().to(device)
    p11_target_model.load_state_dict(p11
        _policy_model.state_dict())

    #initialize pgn network (SL) for each player
    p00_pgn = Policy().to(device)
    p01_pgn = Policy().to(device)
    p10_pgn = Policy().to(device)
    p11_pgn = Policy().to(device)

    #initialize replay buffer for MRL (Reinforcement Learning)
    p00_replay_buffer = ReplayBuffer(200)
    p01_replay_buffer = ReplayBuffer(200)
    p10_replay_buffer = ReplayBuffer(200)
    p11_replay_buffer = ReplayBuffer(200)

    #initialize reservoir buffer for MSL (Supervised Learning)
    p00_reservoir_buffer = ReservoirBuffer(200)
    p01_reservoir_buffer = ReservoirBuffer(200)
    p10_reservoir_buffer = ReservoirBuffer(200)
    p11_reservoir_buffer = ReservoirBuffer(200)

    #Initialize anticipatory parameter η, multi-learning
    p00_state_deque = deque(maxlen=1)
    p00_reward_deque = deque(maxlen=1)
    p00_action_deque = deque(maxlen=1)

    p01_state_deque = deque(maxlen=1)
    p01_reward_deque = deque(maxlen=1)
    p01_action_deque = deque(maxlen=1)

    p10_state_deque = deque(maxlen=1)
    p10_reward_deque = deque(maxlen=1)
    p10_action_deque = deque(maxlen=1)

    p11_state_deque = deque(maxlen=1)
    p11_reward_deque = deque(maxlen=1)
    p11_action_deque = deque(maxlen=1)

    #optimizer for all players (RL)
    p00_rl_optimizer = optim.Adam(p1_policy_model.parameters(), lr=1e-4)
    p01_rl_optimizer = optim.Adam(p2_policy_model.parameters(), lr=1e-4)
    p10_rl_optimizer = optim.Adam(p3_policy_model.parameters(), lr=1e-4)
    p11_rl_optimizer = optim.Adam(p4_polict_model.parameters(), lr=1e-4)

    #optimizer for all players (SL)
    p00_sl_optimizer = optim.Adam(p1_pgn.parameters(), lr=1e-4)
    p01_sl_optimizer = optim.Adam(p2_pgn.parameters(), lr=1e-4)
    p10_sl_optimizer = optim.Adam(p3_pgn.parameters(), lr=1e-4)
    p11_sl_optimizer = optim.Adam(p4_pgn.parameters(), lr=1e-4)

    #p_01 is a teammate, the rest are opponents. Need to initialize this for nfsp
    players = {}
    players['p_00'] = (p00_policy_model, p00_pgn)
    players['p_01'] = (p01_policy_model, p01_pgn)
    players['p_10'] = (p10_policy_model, p10_pgn)
    players['p_11'] = (p11_policy_model, p11_pgn)

    #to determine ordering
    order = ['p_00', 'p_11', 'p_01', 'p_10']

    length_list = []
    p00_reward_list, p00_rl_loss_list, p00_sl_loss_list = [], [], []
    p01_reward_list, p01_rl_loss_list, p01_sl_loss_list = [], [], []
    p10_reward_list, p10_rl_loss_list, p10_sl_loss_list = [], [], []
    p11_reward_list, p11_rl_loss_list, p11_sl_loss_list = [], [], []
    p00_episode_reward, p01_episode_reward, p10_episode_reward, p11_episode_reward = 0, 0, 0, 0

    for episode in n_episodes:

        #create a new environment with a new random bid
        bid_level = random.randint(7,13)
        bid_trump = random.choice(['C', 'D', 'H', 'S', None])
        bid_team = random.choice([0,1])
        env.reset(bid_level, bid_trump, bid_team)

        for r in range(13):
            index = random.randomint(0,3)
            p00_state, p01_state, p10_state, p11_state = None, None, None, None
            p00_action, p01_action, p10_action, p11_action = None, None, None, None

            best = True
            if eta < random.random():
                best = False

            for i in range(4):
                pid = order[index]

                if pid == 'p_00':
                    p00_state, p00_action = take_action(pid, env, prev_p00_state, p00_policy_model, p00_pgn, best)
                elif pid == 'p_01':
                    p01_state, p01_action = take_action(pid, env, prev_p01_state, p01_policy_model, p01_pgn, best)
                elif pid == 'p_10':
                    p10_state, p10_action = take_action(pid, env, prev_p10_state, p10_policy_model, p10_pgn, best)
                elif pid == 'p_11':
                    p11_state, p11_action = take_action(pid, env, prev_p11_state, p11_policy_model, p11_pgn, best)

                index = (index+1) % 4

            (obs00, reward00, done, info) = env.step('p_00')
            (obs01, reward01, done, info) = env.step('p_01')
            (obs10, reward10, done, info) = env.step('p_10')
            (obs11, reward11, done, info) = env.step('p_11')

            p00_state_deque.append(p00_state)
            p01_state_deque.append(p01_state)
            p10_state_deque.append(p10_state)
            p11_state_deque.append(p11_state)

            reward00_tensor = torch.tensor([reward00], device=device)
            reward01_tensor = torch.tensor([reward01], device=device)
            reward10_tensor = torch.tensor([reward00], device=device)
            reward11_tensor = torch.tensor([reward01], device=device)

            next_p00_state = env.get_state('p_00')
            next_p01_state = env.get_state('p_01')
            next_p10_state = env.get_state('p_10')
            next_p11_state = env.get_state('p_11')

            p00_reward_deque.append(reward00_tensor)
            p01_reward_deque.append(reward01_tensor)
            p10_reward_deque.append(reward10_tensor)
            p11_reward_deque.append(reward11_tensor)

            p00_action_deque.append(p00_action)
            p01_action_deque.append(p01_action)
            p10_action_deque.append(p10_action)
            p11_action_deque.append(p11_action)

            #hmmm
            done = True

            #multistep = 1
            if len(p1_state_deque) == 1 or done = True:
                #0.99 is gamma
                n_reward = multi_step_reward(p00_reward_deque, 0.99)
                n_state = p00_state_deque[0]
                n_action = p00_action_deque[0]
                p00_replay_buffer.push(n_state, n_action, n_reward, p00_next_state)

                n_reward = multi_step_reward(p01_reward_deque, 0.99)
                n_state = p01_state_deque[0]
                n_action = p01_action_deque[0]
                p01_replay_buffer.push(n_state, n_action, n_reward, p01_next_state)

                n_reward = multi_step_reward(p10_reward_deque, 0.99)
                n_state = p10_state_deque[0]
                n_action = p10_action_deque[0]
                p10_replay_buffer.push(n_state, n_action, n_reward, p10_next_state)

                n_reward = multi_step_reward(p11_reward_deque, 0.99)
                n_state = p11_state_deque[0]
                n_action = p11_action_deque[0]
                p11_pgn_replay_buffer.push(n_state, n_action, n_reward, p11_next_state)

            if best:
                p00_reservoir_buffer.push(p00_state, p00_action)
                p01_reservoir_buffer.push(p01_state, p01_action)
                p10_reservoir_buffer.push(p10_state, p10_action)
                p11_reservoir_buffer.push(p11_state, p11_action)

            p00_state = p00_next_state
            p01_state = p01_next_state
            p10_state = p10_next_state
            p11_state = p11_next_state

            p00_episode_reward += reward00_tensor
            p01_episode_reward += reward01_tensor
            p10_episode_reward += reward10_tensor
            p11_episode_reward += reward11_tensor

            #need to finish rest
            if done:
                p00_reward_list.append(p00_episode_reward)
                p01_reward_list.append(p01_episode_reward)
                p10_reward_list.append(p10_episode_reward)
                p11_reward_list.append(p11_episode_reward)

                p00_state_deque.clear()
                p00_reward_deque.clear()
                p00_action_deque.clear()

                p01_state_deque.clear()
                p01_reward_deque.clear()
                p01_action_deque.clear()

                p10_state_deque.clear()
                p10_reward_deque.clear()
                p10_action_deque.clear()

                p11_state_deque.clear()
                p11_reward_deque.clear()
                p11_action_deque.clear()

            if (len(p00_replay_buffer) > rl_start and len(p00_reservoir_buffer) > sl_start and episode % train_freq == 0):
                #updating best responses
                loss = rl_loss(p00_policy_model, p00_target_model, p00_replay_buffer, p00_rl_optimizer)
                p00_rl_loss_list.append(loss.item())

                loss = rl_loss(p01_policy_model, p01_target_model, p01_replay_buffer, p01_rl_optimizer)
                p01_rl_loss_list.append(loss.item())

                loss = rl_loss(p10_policy_model, p10_target_model, p10_replay_buffer, p10_rl_optimizer)
                p10_rl_loss_list.append(loss.item())

                loss = rl_loss(p11_policy_model, p11_target_model, p11_replay_buffer, p11_rl_optimizer)
                p11_rl_loss_list.append(loss.item())

                #updating average strategy
                loss = sl_loss(p00_policy, p00_reservoir_buffer, p00_sl_optimizer)
                p00_sl_loss_list.append(loss.item())

                loss = sl_loss(p01_policy, p01_reservoir_buffer, p01_sl_optimizer)
                p01_sl_loss_list.append(loss.item())

                loss = sl_loss(p10_policy, p10_reservoir_buffer, p10_sl_optimizer)
                p10_sl_loss_list.append(loss.item())

                loss = sl_loss(p11_policy, p11_reservoir_buffer, p11_sl_optimizer)
                p11_sl_loss_list.append(loss.item())


            if episode % 10 == 0:
                p00_target_model.load_state_dict(p00_policy_model.state_dict())
                torch.save(p00_policy_model.state_dict(), "models/p00-network-{}.pth".format(episode))
                p01_target_model.load_state_dict(p01_policy_model.state_dict())
                torch.save(p01_policy_model.state_dict(), "models/p01-network-{}.pth".format(episode))
                p10_target_model.load_state_dict(p10_policy_model.state_dict())
                torch.save(p10_policy_model.state_dict(), "models/p10-network-{}.pth".format(episode))
                p11_target_model.load_state_dict(p11_policy_model.state_dict())
                torch.save(p11_policy_model.state_dict(), "models/p11-network-{}.pth".format(episode))

                torch.save(p00_pgn.state_dict(), "policies/p00-network-{}.pth".format(episode))
                torch.save(p01_pgn.state_dict(), "policies/p01-network-{}.pth".format(episode))
                torch.save(p10_pgn.state_dict(), "policies/p10-network-{}.pth".format(episode))
                torch.save(p11_pgn.state_dict(), "policies/p11-network-{}.pth".format(episode))

            #save models
            if episode % 10000 == 0:
                p00_reward_list.clear(), p01_reward_list.clear(), p10_reward_list.clear(), p11_reward_list.clear()
                p00_rl_loss_list.clear(), p01_rl_loss_list.clear(), p10_rl_loss_list.clear(), p11_rl_loss_list.clear()
                p00_sl_loss_list.clear(), p01_sl_loss_list.clear(), p10_sl_loss_list.clear(), p11_sl_loss_list.clear()




if __name__ == '__main__':
    env = gym.make('contract_bridge:contract-bridge-v0')
    env.initialize(8, None, 0)
    train(env)
