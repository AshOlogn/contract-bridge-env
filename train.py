import gym
from .env import BridgeEnv
from .agent import Agent

env = gym.make('BridgeEnv')



def train(args):
	episodes = args.episodes
	reward = 0
	done = False

	agent = Agent(env.action_space)

	for i in range(episodes):
		init_observation = env.reset()

		while True:
			action = agent.act(init_observation, reward, done)
			ob, reward, done, _ = env.step(action)

			#once you get state info for that action. Insert it as an input to the model.
            if done:
                break





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=100)
    args = parser.parse_args()
    train(args)