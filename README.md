# CS 394R: Reinforcement Learning Final Project

Our repository is structured as follows:
```
├── agents
│   ├── agent.py
├── contract_bridge
│   └── envs
│       ├── bridge_trick_taking.py
│       ├── deck.py
├── nfsp
│   ├── buffers.py
│   ├── nn.py
├── trained_agents
│   ├── dqn-random.pth
│   ├── dqn-self-play.pth
│   └── dqn-smart.pth
├── train_dqn.py
└── train.py
├── test.py
```

The following are descriptions of the most important files/directories of interest:
* ``agent.py``: This contains our implementations of a random agent, rule-based agent, and the DQN-based agent.
* ``bridge_trick_taking.py``: Contains our implementation of the multi-agent Bridge game environment (mostly follows the OpenAI Gym standards but deviates a bit to support multiple agents)
* ``nn.py``: Contains our implementation of the DQN
* ``train.py``: Script that trains a DQN against either a random agent or a rule-based agent and serializes the network.
* ``train_dqn``		: Script that implements basic self-play by jointly training 4 DQNs
* ``test.py``: Script that tests teams of trained agents against each other over user-specified number of rounds (further instructions below)
* ``trained_agents``: Directory containing pre-trained DQN models that can be used by ``test.py`` out of the box.

## How to Run Our Code 
#### Gym Environment Installation
Before running **any other code**, our Gym environment must be installed locally with the command ``pip install -e .`` run from the root of this repository.

#### Training
``train.py`` takes a single command-line argument, ``random`` or ``smart``, which specifies whether a DQN should be trained against a random agent or rule-based agent. 

This script then saves a serialized model in a directory generated called ``models`` and generates logs of a sliding window average win rate of  the DQN being trained against the opponent agent in a directory called ``logs``.

``train_dqn.py`` takes no command line arguments. It jointly trains 4 DQNs representing the 4 players in Bridge.

This script then saves 4 serialized models in a directory generated called ``models/dqn`` and generates logs of the observed loss of the neural network every 1000 generations. The 4 models are denoted as p_00, p_01 (team 0) and p_10, p_11 (team 1).

#### Testing
``test.py`` takes 3 command-line arguments representing the type of agent each team should play with.  The _first two_ arguments can each be one of the following:
* random
* smart: rule-based agent
* dqn-random: DQN trained against random agent
* dqn-smart: DQN trained against rule-based agent
* dqn-self-play: DQN jointly trained with 3 other DQNs

The dqn commands use pre-trained sample networks in the directory ``trained_agents``. If you train your own DQNs, rename them appropriately and place them in that directory.

The _final_ argument is the number of rounds you want the agents to play. The output in stdout is then the win percentage of each team.

Sample commands: ``python3 test.py random random 1000`` 
Sample response:
```
Team 0 wins: 503, Team 1 wins: 497
Team 0 percentage: 0.50, Team 2 percentage: 0.50
```

## Link to Our Video

[https://tinyurl.com/rl-final-project](https://tinyurl.com/rl-final-project)
