#!/usr/bin/env python
# coding: utf-8

# In[129]:


import torch
import torch.nn as nn
from collections import namedtuple
import gym
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


# In[130]:


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
MAX_ITER = 10000000


# In[131]:


class Net(nn.Module):
    def __init__(self, num_features:int, hidden_size:int, num_actions:int):
        super(Net, self).__init__()
        # Define two linear layers
        self.linear1 = nn.Linear(in_features=num_features, out_features=hidden_size)
        self.linear2 = nn.Linear(in_features=hidden_size, out_features=num_actions)

        # Define a ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)  # First linear layer
        x = self.relu(x)     # ReLU activation
        x = self.linear2(x)  # Second linear layer
        return x


# In[132]:


Episode = namedtuple('Episode', field_names=['total_rewards', 'steps'])
episodeStep = namedtuple('episodeStep', field_names=['state', 'action'])


# In[133]:


def create_episode(env:gym.Env, net:Net):
    net.eval()
    total_rewards = 0
    steps = []
    sm = nn.Softmax(dim=0)
    
    current_state = env.reset()[0]
    
    while True:
        current_state_tensor = torch.FloatTensor(current_state)
        action_prob = sm(net(current_state_tensor))
        action_prob = action_prob.detach().numpy()
        action = np.random.choice(env.action_space.n, p=action_prob)
        
        next_state, reward, terminated, _, info = env.step(action=action)
        total_rewards += reward
        current_step = episodeStep(state=current_state, action=action)
        steps.append(current_step)
        
        if terminated:
            e = Episode(total_rewards=total_rewards, steps=steps)
            return e
            
        
        current_state = next_state
    


# In[134]:


def create_batch(env:gym.Env, net:Net, batch_size:int):
    batch = []
    for i in range(batch_size):
        episode = create_episode(env, net)
        batch.append(episode)
        
    return batch
    


# In[135]:


def filter_batch(batch, percentile):
    rewards = list(map(lambda s:s.total_rewards, batch))
    reward_percentile = np.percentile(rewards, percentile)
    mean_reward = np.mean(rewards)
    
    training_states = []
    training_actions = []
    for total_rewards, steps in batch:
        if total_rewards< reward_percentile:
            continue
        training_states.extend(map(lambda step: step.state, steps))
        training_actions.extend(map(lambda step: step.action, steps))
        
    return training_states, training_actions, reward_percentile, mean_reward
        
        
        


# In[136]:


if __name__=="__main__":
    env = gym.make("CartPole-v0")
    
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n
    net = Net(num_features=num_features, hidden_size=HIDDEN_SIZE, num_actions=num_actions)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(params = net.parameters(), lr = 0.01)
    
    writer = SummaryWriter()
    for i in range(MAX_ITER):
        batch = create_batch(env, net, BATCH_SIZE)
        training_states, training_actions, reward_percentile, mean_reward = filter_batch(batch=batch, percentile=PERCENTILE)
        training_states = torch.FloatTensor(training_states)
        training_actions = torch.LongTensor(training_actions)
        # Zero the parameter gradients
        net.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = net(training_states)
        outputs = outputs.float()
        loss = criterion(outputs, training_actions)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        print('Current iteration: {}, mean reward: {}'.format(i, mean_reward))
        
        writer.add_scalar("loss", loss.item(), i)
        writer.add_scalar("reward_bound", reward_percentile, i)
        writer.add_scalar("mean_reward", mean_reward, i)
        if mean_reward > 199:
            break
        
    writer.close()
        


# In[137]:


# env = gym.make("CartPole-v0")
# current_state = env.reset()[0]
# current_state = torch.FloatTensor(current_state)


# In[138]:


# num_features = env.observation_space.shape[0]
# num_actions = env.action_space.n
# net = Net(num_features=num_features, hidden_size=HIDDEN_SIZE, num_actions=num_actions)


# In[ ]:




