# https://www.section.io/engineering-education/building-a-reinforcement-learning-environment-using-openai-gym/
# https://colab.research.google.com/drive/1oBe07b28h9GCBy_bKtLJisC98mayDcwn?usp=sharing#scrollTo=ns6duy9JDP3S

import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random


class CustomEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # self.state = 38 + random.randint(-3, 3)
        self.state = random.randint(0, 100)
        self.shower_length = 60

    def display_all(self):
        print(type(self.action_space), self.action_space)
        print(type(self.observation_space), self.observation_space)
        print(type(self.state), self.state)
        print(type(self.shower_length), self.shower_length)

    def display_observation_space(self):
        print(self.observation_space)

    def step(self, action: int):
        self.state += action - 1 # [0, 1, 2] -> [-1, 0, 1]
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
        # self.state = 38 + random.randint(-3, 3)
        self.state = random.randint(0, 100)
        self.shower_length = 60
        return self.state


env = CustomEnv()

env.observation_space.sample()
env.action_space.sample()

episodes = 1  # 20 shower episodes
for episode in range(1, episodes + 1):

    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        print(f'n_state:{n_state} done:{done} action:{action} reward:{reward} score:{score+reward}')
        env.display_observation_space()

        # env.display_all()
        # print(action)
        score += reward


    print(f'Episode:{episode} Score:{score}')
    print()
    print()

