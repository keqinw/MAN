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


class MAN():
    # this class is for the MAN, which used to predict the position of cube
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
        self.action_n = self.env.action_space
        self.obs_n = self.env.observation_space
        self.target_nets = []
        self.evaluate_nets = []
        self.optimizers = []

        previous_action = 0
        for num_action in self.action_n:
            self.target_nets.append(FullyConnectedModel(state_dim=self.obs_n+previous_action,action_dim=num_action))
            self.evaluate_nets.append(FullyConnectedModel(state_dim=self.obs_n+previous_action,action_dim=num_action))
            
            self.target_nets[-1].load_state_dict(self.evaluate_nets[-1].state_dict())
            self.target_nets[-1].eval()

            self.optimizers.append(torch.optim.Adam(self.evaluate_nets[-1].parameters(),lr = self.lr))
            previous_action += 1

    def save_model_weights(self):
    # Helper function to save your model / weights. 
        for i,model in enumerate(self.evaluate_nets):
            self.path = os.path.join(self.logdir, "model_man_")
            torch.save(model.state_dict(), self.path + str(i))
        return self.path

    def load_model(self,path):
    # Helper function to load an existing model.
        for i in range(self.env.n):
            self.evaluate_nets[i].load_state_dict(torch.load(path +str(i)))

    def save_best_model(self):
    # Helper function to save the best model.
        for i,model in enumerate(self.evaluate_nets):
            self.path = os.path.join(self.logdir, "model_man_" + str(i) + '_best')
            torch.save(model.state_dict(), self.path)
        return self.path

class Replay_Memory():
    def __init__(self, memory_size=100000, burn_in=10000):

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        self.memory_size = memory_size
        self.burn_in = burn_in
        self.mem_pool = collections.deque([], maxlen=self.memory_size)

    def sample_batch(self, batch_size=256):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
        return random.sample(self.mem_pool, batch_size)

    def append(self, transition):
    # Appends transition to the memory.
        self.mem_pool.append(transition)

