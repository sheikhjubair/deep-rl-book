#!/usr/bin/env python
# coding: utf-8

# In[98]:


import gym


# In[99]:


if __name__ =="__main__":
    env = gym.make('CartPole-v0')
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


# In[100]:


print("Episode done in {} steps, total rewards {}".format(total_steps, total_reward))

