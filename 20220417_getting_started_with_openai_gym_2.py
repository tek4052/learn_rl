# https://blog.paperspace.com/getting-started-with-openai-gym/

import gym
import matplotlib.pyplot as plt
import time
import datetime

start = datetime.datetime.now()

env = gym.make('MountainCar-v0')

# Observation and action space
print(f"type(env.observation_space): {type(env.observation_space)}")
# Box(n,) corresponds to the n-dimensional continuous space
print("Upper Bound for Env Observation", env.observation_space.high)
print("Lower Bound for Env Observation", env.observation_space.low)
print(f"type(env.action_space): {type(env.action_space)}")
# a discrete space with [0.....n-1] possible values
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))


# # reset the environment and see the initial observation
# obs = env.reset()
# print("The initial observation is {}".format(obs))
#
# # Sample a random action from the entire action space
# random_action = env.action_space.sample()
#
# # Take the action and get the new observation space
# new_obs, reward, done, info = env.step(random_action)
# print("The new observation is {}".format(new_obs))
#
# # env.render(mode = "human")
# env_screen = env.render(mode='rgb_array')
# plt.imshow(env_screen)


# Number of steps you run the agent for
num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs)
    action = env.action_space.sample()

    # apply the action
    obs, reward, done, info = env.step(action)

    # Render the env
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    # time.sleep(0.001)
    # print(f"step: {step} done: {reward} reward: {reward}")

    # If the episode is up, then start another one
    if done:
        env.reset()

# Close the env
env.close()

end = datetime.datetime.now()

print(f"Duration: {end - start}")
