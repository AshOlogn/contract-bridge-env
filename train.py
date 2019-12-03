import gym
import torch 
import torch.nn as nn
from .env import BridgeEnv
from .agent import DQNAgent
from .DQN import DeepQNetwork

env = gym.make('BridgeEnv')



def train(args):

	batch_size = 10000
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10 # update target network after 10 episodes
    steps_done = 0

	episodes = args.episodes
	reward = 0
	done = False

	agent = DQNAgent(env.action_space)

	for i in range(episodes):
		init_observation = env.reset()

		while True:
			action = agent.act(init_observation, reward, done)
			ob, reward, done, _ = env.step(action)

			#once you get state info for that action(ob). Insert it as an input to the model.

            if done:
                break





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=100)
    args = parser.parse_args()
    train(args)