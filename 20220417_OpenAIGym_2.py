# https://www.section.io/engineering-education/building-a-reinforcement-learning-environment-using-openai-gym/
# https://colab.research.google.com/drive/1oBe07b28h9GCBy_bKtLJisC98mayDcwn?usp=sharing

import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import datetime

class CustomEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60

    def step(self, action):
        self.state += action - 1
        self.shower_length -= 1

        if 37 <= self.state <= 39:
            reward = 1
        else:
            reward = -1

        if self.shower_length <= 0:
            done = True
        else:
            done = False

        info = {}

        # Return step information
        return self.state, reward, done, info

    def reset(self):
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        return self.state

env = CustomEnv()
env.observation_space.sample()
env.action_space.sample()

states = env.observation_space.shape
actions = env.action_space.n

def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(states, actions)
model.summary()

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model,
                   memory=memory,
                   policy=policy,
                   nb_actions=actions,
                   nb_steps_warmup=10,
                   target_model_update=1e-2)
    return dqn

start = datetime.datetime.now()

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=60000, visualize=False, verbose=1)

results = dqn.test(env, nb_episodes=150, visualize=False)
print(np.mean(results.history['episode_reward']))

end = datetime.datetime.now()

print(f"Duration: {end - start}")
