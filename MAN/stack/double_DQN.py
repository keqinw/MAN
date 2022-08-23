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
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_dim)

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

    def __init__(self, env, lr, logdir='./model'):
    # Define your network architecture here. It is also a good idea to define any training operations
    # and optimizers here, initialize your variables, or alternately compile your model here.
        pass
        self.env = env
        self.lr = lr
        self.logdir = logdir
        self.action_n = env.action_space.n * 4
        self.obs_n = self.env.observation_space.shape[0] 
        self.model = FullyConnectedModel(state_dim=self.obs_n,action_dim=self.action_n)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.criterion = torch.nn.MSELoss()
        # 这里是一步比较重要的，到底是放在dqn里面还是外面，好像是有区别的，需要继续思考。这里先放在外面。

    def save_model_weights(self):
    # Helper function to save your model / weights. 
        self.path = os.path.join(self.logdir, "model_double")
        torch.save(self.model.state_dict(), self.path)
        return self.path

    def load_model(self,path):
    # Helper function to load an existing model.
        return self.model.load_state_dict(torch.load(path))

    def save_best_model(self,c=None):
        if c is not None:
            self.path = os.path.join(self.logdir, "best_model_double_"+str(c))
        else:
            self.path = os.path.join(self.logdir, "best_model_double")
        torch.save(self.model.state_dict(), self.path)
        return self.path
    


class Replay_Memory():

    def __init__(self, memory_size=5000, burn_in=50,state_size = 14, action_size = 1):

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.memory_size = memory_size
        self.burn_in = burn_in
        # self.mem_pool = collections.deque([], maxlen=self.memory_size)
        self.state = torch.empty(self.memory_size, state_size, dtype=torch.float)
        self.action = torch.empty(self.memory_size, action_size, dtype=torch.long)
        self.reward = torch.empty(self.memory_size, dtype=torch.float)
        self.next_state = torch.empty(self.memory_size, state_size, dtype=torch.float)
        self.done = torch.empty(self.memory_size, dtype=torch.int)
        self.count = 0
        self.idx = 0

    def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
        N = min(self.memory_size, self.count)
        indices = np.random.choice(N, batch_size ,replace = True)
        batch = (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.done[indices]
        )
        return batch

    def add(self, transition):
    # Appends transition to the memory.
        state, action, reward, next_state, done = transition
        # store transition in the buffer
        self.state[self.idx] = torch.as_tensor(state)
        self.action[self.idx] = torch.as_tensor(action)
        self.reward[self.idx] = torch.as_tensor(reward)
        self.next_state[self.idx] = torch.as_tensor(next_state)
        self.done[self.idx] = torch.as_tensor(done)
        self.idx = (self.idx + 1) % self.memory_size 

        self.count += 1



