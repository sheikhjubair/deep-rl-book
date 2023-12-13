#!/usr/bin/env python
# coding: utf-8

# In[8]:


import gym
from typing import TypeVar
import random


# In[9]:


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon
        
    def action(self, action):
        if random.random() < self.epsilon:
            print('Random Action')
            action = self.env.action_space.sample()
            
        return action


# In[10]:


if __name__ =="__main__":
    env = RandomActionWrapper(gym.make('CartPole-v0'))
    obs = env.reset()
    
    total_reward = 0
    total_steps = 0
    
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, _, info = env.step(action=action)
        
        total_reward += reward
        total_steps += 1
        
        if terminated:
            break


# In[11]:


print("Episode done in {} steps, total rewards {}".format(total_steps, total_reward))

