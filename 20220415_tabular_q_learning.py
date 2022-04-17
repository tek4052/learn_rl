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
MAX_EPISODES = 100000 + 1
PAUSE_TIME = 0.0 # 1
FRESH_TIME = 0.0 # 0.1

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), # q_table initial values
        columns=actions,
    )
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]

    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else: # act greedy with 90% probability
        action_name = ACTIONS[state_actions.argmax()]

    return action_name

def get_env_feedback(S, A):
    R = 0

    if A == 'right':
        if S == N_STATES - 2: # terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1

    else: # left
        if S == 0:
            S_ = S # reach the wall
        else:
            S_ = S - 1
    
    return S_, R

def update_env(S, episode, step):
    env_list = ['-']*(N_STATES-1) + ['T']

    if S == "terminal":
        interaction = f"Episode {episode:<3} {'totalsteps'} {step-1}"
        print(interaction)
        print()
        print()
        time.sleep(PAUSE_TIME)

    else:
        env_list[S] = 'o'
        interaction = ' '.join(env_list)
        interaction = f"Episode {episode:<3} {'step'} {step:<3}\t{interaction}"
        # print(interaction)
        time.sleep(FRESH_TIME)

q_table = build_q_table(N_STATES, ACTIONS)

for episode in range(1, MAX_EPISODES, 1):

    step = 0
    S = 0
    is_terminated = False

    print(q_table.to_markdown())

    update_env(S, episode, step)

    while not is_terminated:

        #### Choose an action
        A = choose_action(S, q_table)
        S_, R = get_env_feedback(S, A)

        q_predict = q_table.at[S, A]
            
        if S_ != "terminal":
            q_target = R + LAMBDA * q_table.iloc[S_, :].max() # next state is not terminal
        else:
            q_target = R # next state is terminal
            is_terminated = True

        #### Update Q table
        q_table.at[S, A] += ALPHA * (q_target - q_predict)

        #### Move to next state
        S = S_

        #### Increate step
        step = step + 1

        #### Display environment
        update_env(S, episode, step)
