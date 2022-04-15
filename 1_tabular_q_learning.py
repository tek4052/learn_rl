#### https://www.youtube.com/watch?v=OkGFJE_XDzI&list=PLXO45tsB95cJYKCSATwh1M4n8cUnUv6lT&index=6

import numpy as np
import pandas as pd
import time


np.random.seed(2)


N_STATES = 6 # number of states
ACTIONS = ['left', 'right'] # possible actions at each states
EPSILON = 0.9 # greeday policy
ALPHA = 0.1 # learning rate
LAMBDA = 0.9 # discount factor
MAX_EPISODES = 13
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # q_table initial values
        columns=actions,
    )
    return table

print(build_q_table(N_STATES, ACTIONS).to_markdown())