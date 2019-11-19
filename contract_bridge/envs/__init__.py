from gym.envs.registration import register

register(
    id='gym_gas_locator-v0',
    entry_point='contract_bridge.envs:BridgeEnv',
)