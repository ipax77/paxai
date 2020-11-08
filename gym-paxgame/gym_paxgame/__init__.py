from gym.envs.registration import register

register(
    id='paxgame-v0',
    entry_point='gym_paxgame.envs:PaxGameEnv',
)
