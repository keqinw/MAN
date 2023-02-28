# here is the neural network structure of DQN
# this is the test version, which the input is the height list + cube list, not the encoded image.
# just test if the method work or not.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import collections
import random

# this is a full connection layer, both DQN-I and DQN-II share the same structure (first)
class FullyConnectedModel(torch.nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # here the state_dim is (14,1)
        # no image, therefore, no need for convolution layer
        # the structure might be modified in the future
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, action_dim)

        #initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQN():
    # this class is for the DQN-I, which used to predict the type of cube
    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, n,lr, logdir='./model'):
    # Define your network architecture here. It is also a good idea to define any training operations
    # and optimizers here, initialize your variables, or alternately compile your model here.
        pass
        self.env = env
        self.lr = lr
        self.logdir = logdir
        self.action_n = 3**n
        self.obs_n = 2*n
        self.model = FullyConnectedModel(state_dim=self.obs_n,action_dim=self.action_n)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.criterion = torch.nn.MSELoss()
        # 这里是一步比较重要的，到底是放在dqn里面还是外面，好像是有区别的，需要继续思考。这里先放在外面。

    def save_model_weights(self):
    # Helper function to save your model / weights. 
        self.path = os.path.join(self.logdir, "model_vanilla")
        torch.save(self.model.state_dict(), self.path)
        return self.path

    def load_model(self,path):
    # Helper function to load an existing model.
        return self.model.load_state_dict(torch.load(path))

    def save_best_model(self):
        self.path = os.path.join(self.logdir, "best_model_vanilla")
        torch.save(self.model.state_dict(), self.path)
        return self.path
    


class Replay_Memory():

    def __init__(self, memory_size=100000, burn_in=10000):
    # def __init__(self, memory_size=5000, burn_in=1000):
    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.memory_size = memory_size
        self.burn_in = burn_in
        self.mem_pool = collections.deque([], maxlen=self.memory_size)

    def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
        return random.sample(self.mem_pool, batch_size)

    def append(self, transition):
    # Appends transition to the memory.
        self.mem_pool.append(transition)

