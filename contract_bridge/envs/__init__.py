from gym.envs.registration import register

register(
    id='contract-bridge-v0',
    entry_point='contract_bridge.envs:BridgeEnv',
)