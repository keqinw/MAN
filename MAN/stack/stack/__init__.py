from gym.envs.registration import register

register(
    id='Stack-v0', 
    entry_point='stack.envs:StackEnv_v0'
)

register(
    id='Stack-v1', 
    entry_point='stack.envs:StackEnv_v1'
)
