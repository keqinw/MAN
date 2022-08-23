# here is the vanilla dqn, which share the same structure with others, but with larger output space.
# Used to compare the performance with my own algorithm.

import numpy as np
from sqlalchemy import over
import torch
import gym
import stack
import matplotlib.pyplot as plt
import sys
import tqdm
import argparse
from DQN import Replay_Memory
from DQN import DQN
import math
import pandas as pd

heights = torch.tensor([ [0,0.025,0], [0.025,0.025,0.025], [0.05,0.05,0.05], [0.075,0.075,0.075] ])
SAVE_STR = 'vanilla'
class Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, lr, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        pass
        self.env = gym.make(environment_name)
        self.environment_name = environment_name
        self.replay_memory = Replay_Memory()
        self.current_network = DQN(self.env,lr)
        self.target_network = DQN(self.env,lr)
        # self.target_network.path = self.current_network.save_model_weights()
        # self.target_network.load_model()
        self.target_network.model.eval()

        self.report_freq = 500 #50
        self.epsilon_start = 1
        self.epsilon_end = 0.1
        self.epsilon_decay = 10000
        self.gamma = 0.99
        self.batch_size = 32
        self.replay_memory = Replay_Memory()
        self.render = render
        self.c = 0
        self.burn_in_memory()
        self.optimizer = torch.optim.Adam(self.current_network.model.parameters(),lr = self.current_network.lr)
        self.tau = 0.005

    def soft_update(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)

    def epsilon_greedy_policy(self, q_values,over_ride = None):
        # Creating epsilon greedy probabilities to sample from.
        # Here we use the soft-greedy method
        if over_ride is not None:
            epsilon = over_ride
        else:    
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.c / self.epsilon_decay)
        
        if np.random.rand() >= epsilon:
            # print('greedy',self.greedy_policy(q_values)  )
            return self.greedy_policy(q_values)           
        else:
            # print('epsilon:',self.env.action_space.sample())
            return np.random.choice(self.env.cube_pool), self.env.action_space.sample()

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        sorted, indices = torch.sort(q_values, descending=True)
        for index in indices.tolist()[0]:
            type,action = divmod(index,14)
            if type in self.env.cube_pool:
                return type,action

    def train(self,num_step):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        pass
    
        # self.current_network.model.train()
        # c = 0
        reward_mean, reward_std, max_height_mean, max_height_std, bumpiness_mean, bumpiness_std = test(self.environment_name, self.current_network.model)
        reward_means = [reward_mean]
        reward_stds = [reward_std]
        max_height_means = [max_height_mean]
        max_height_stds = [max_height_std]
        bumpiness_means = [bumpiness_mean]
        bumpiness_stds = [bumpiness_std]
        total_loss = 0
        
        state = torch.tensor(self.env.reset()).view(1,-1).float()

        for i in range(num_step):
            self.optimizer.zero_grad()

            q_values = self.current_network.model(state)
            action_store = torch.argmax(q_values).item()
            index,action = self.epsilon_greedy_policy(q_values)

            next_state,reward,done,_ = self.env.step([index,action])
            transition = [state,14*index+action,reward,next_state,done]
            self.replay_memory.add(transition)
            mini_batch = self.replay_memory.sample_batch(self.batch_size) #list

            sample_state = mini_batch[0]
            sample_action = mini_batch[1]
            sample_reward = mini_batch[2]
            sample_next_state = mini_batch[3]
            sample_done = mini_batch[-1]

            q_next = torch.max(self.target_network.model(sample_next_state).detach()) 
            target = sample_reward + self.gamma * q_next * (1 - sample_done)

            eval = torch.gather(self.current_network.model(sample_state),1,sample_action).squeeze(1)

            loss = torch.mean((target - eval)**2 )
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.soft_update(self.target_network.model, self.current_network.model)

            self.c += 1
            total_loss += loss.item()

            if self.c % self.report_freq == 0 :
                reward_mean, reward_std, max_height_mean, max_height_std, bumpiness_mean, bumpiness_std = test(self.environment_name, self.current_network.model)
                reward_means.append(reward_mean)
                reward_stds.append(reward_std)
                max_height_means.append(max_height_mean)
                max_height_stds.append(max_height_std)
                bumpiness_means.append(bumpiness_mean)
                bumpiness_stds.append(bumpiness_std)

                eps = max(self.epsilon_end,self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.c / self.epsilon_decay)
                print(f"Step: {self.c}, Reward mean: {reward_mean:.2f}, Reward std: {reward_std:.2f}, Max_height: {max_height_mean:.2f}, Bumpiness: {bumpiness_mean:.2f}, Loss: {total_loss / self.c:.4f}, Eps: {eps:.2f}")
                if reward_means[-1] == max(reward_means):
                    path = self.current_network.save_best_model()
            # if self.c % (self.report_freq*10) == 0 :  
            #     path = self.current_network.save_best_model(self.c)
            if not done:
                state = torch.tensor(next_state).view(1,-1).float()
            else:
                state = torch.tensor(self.env.reset()).view(1,-1).float()

        return reward_means,max_height_means,bumpiness_means

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        pass
        num = 0
        current_state = self.env.reset()
        
        while num < self.replay_memory.burn_in:
            
            action = int(self.env.action_space.sample())
            index = np.random.choice(self.env.cube_pool)
            next_state, reward, done, _ = self.env.step([index,action])
            # print(torch.tensor(current_state).view(1,-1).float())
            # print(torch.tensor(action).view(1).long())

            transition = [current_state,14*index+action,reward,next_state,done]

            self.replay_memory.add(transition)
            num += 1
            if done:
                current_state = self.env.reset()
            if not done:
                current_state = next_state

def test(env_name, model, model_file=None):
    # Evaluate the performance of your agent over 100 episodes, by calculating average cummulative rewards (returns) for the 100 episodes.
    # Here you need to interact with the environment, irrespective of whether you are using a memory.
    pass
    env = gym.make(env_name)
    with torch.no_grad():
        test_trial = 10
        epsilon = -1
        # self.select_network.model.eval()
        # self.evaluate_network.model.eval()

        total_rewards = []
        total_max_heights = []
        total_bumpiness = []
        
        for _ in range(test_trial):
            state = torch.tensor(env.reset()).view(1,-1).float()
            
            rewards = 0
            while True:
                q_values = model(state)
                sorted, indices = torch.sort(q_values, descending=True)
                for index in indices.tolist()[0]:
                    type,action = divmod(index,14)
                    if type in env.cube_pool:
                        action_I,action_II = type,action
                        break
                next_state, reward, done, _ = env.step([action_I,action_II])
                next_state = torch.tensor(next_state).view(1,-1).float()
                rewards += reward
                state = next_state
                if done:
                    # state = torch.tensor(self.env.reset()).view(1,-1).float()
                    total_max_heights.append(torch.max(next_state).item())

                    virance = 0
                    for i in range(next_state.size(1)-1):
                        virance += torch.abs(next_state[0,i+1] - next_state[0,i])
                    total_bumpiness.append(virance)
                    break

            total_rewards.append(rewards)

        reward_mean = np.mean(total_rewards)
        reward_std = np.std(total_rewards)

        max_height_mean = np.mean(total_max_heights)
        max_height_std = np.std(total_max_heights)  

        bumpiness_mean = np.mean(total_bumpiness)
        bumpiness_std = np.std(total_bumpiness)

    return reward_mean, reward_std, max_height_mean, max_height_std, bumpiness_mean, bumpiness_std

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default="Stack-v1")
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=3e-5)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env
    lr = args.lr
    num_trails = 10
    num_step = 20000
    render = args.render

    # env = gym.make(args.env)
    # ob = env.reset()


    reward_means_total = []
    max_height_means_total = []
    bumpiness_means_total = []

    for trail in tqdm.tqdm(range(num_trails)):
        reward_means = []
        reward_stds = []
        agent = Agent(environment_name,lr,render=render)
        reward_means,max_height_means,bumpiness_means = agent.train(num_step)
        reward_means_total.append(reward_means)
        max_height_means_total.append(max_height_means)
        bumpiness_means_total.append(bumpiness_means)

    reward_means_total = np.array(reward_means_total)
    max_height_means_total = np.array(max_height_means_total)
    bumpiness_means_total = np.array(bumpiness_means_total)

    reward_mean = reward_means_total.mean(axis=0)
    reward_std = reward_means_total.std(axis=0)
    max_height_mean = max_height_means_total.mean(axis=0)
    max_height_std = max_height_means_total.std(axis=0)
    bumpiness_mean = bumpiness_means_total.mean(axis=0)
    bumpiness_std = bumpiness_means_total.std(axis=0)

    # # save the data in a csv file for plotting later on
    pd.DataFrame(reward_mean).to_csv('./data/' + SAVE_STR + '_reward_mean.csv', header=None, index=None)
    pd.DataFrame(reward_std).to_csv('./data/' + SAVE_STR + '_reward_std.csv', header=None, index=None)
    pd.DataFrame(max_height_mean).to_csv('./data/' + SAVE_STR + '_max_height_mean.csv', header=None, index=None)
    pd.DataFrame(max_height_std).to_csv('./data/' + SAVE_STR + '_max_height_std.csv', header=None, index=None)
    pd.DataFrame(bumpiness_mean).to_csv('./data/' + SAVE_STR + '_bumpiness_mean.csv', header=None, index=None)
    pd.DataFrame(bumpiness_std).to_csv('./data/' + SAVE_STR + '_bumpiness_std.csv', header=None, index=None)

    # plt.plot(reward_means)
    # plt.show()
    # plt.savefig("./plots/reward_"+SAVE_STR)

if __name__ == '__main__':
    main(sys.argv)


