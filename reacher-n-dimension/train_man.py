import numpy as np
import torch
from reacher.reacher_env import ReacherEnv_v1
import matplotlib.pyplot as plt
import sys
import tqdm
import argparse
from man import Replay_Memory
from man import MAN
import math
import pandas as pd
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MAN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q & V Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, num_dimension, lr, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.n = num_dimension
        self.env = ReacherEnv_v1(self.n)
        self.replay_memory = Replay_Memory()
        self.agent = MAN(self.env,lr)

        self.update = 1000
        self.epsilon_start = 0.9
        self.epsilon_end = 0.1
        self.epsilon_decay = 10000
        self.gamma = 0.99
        self.batch_size = 32
        self.render = render
        self.c = 0
        self.direction = torch.tensor([-1,0,1])
        self.burn_in_memory()
        

    def epsilon_greedy_policy(self, ob,train=True):
        # Creating epsilon greedy probabilities to sample from.
        # Here we use the soft-greedy method
        pass
        if train: 
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.c / self.epsilon_decay)
        if not train:
            epsilon = 0.05
        if np.random.rand() >= epsilon:
            return self.greedy_policy_Q(ob)           
        else:
            return np.random.choice(self.env.action_space[0],self.env.n)

    def greedy_policy_Q(self, ob):
        # Creating greedy policy for test time.
        ob = torch.tensor(ob,dtype=torch.float32)
        actions = np.array([torch.argmax(self.agent.evaluate_nets[0](ob).detach()).view(-1).item()])
        for i in range(1, self.env.n):
            ob = torch.cat((ob,torch.tensor(actions[-1],dtype=torch.float32).view(1)))
            actions = np.hstack((actions,torch.argmax(self.agent.evaluate_nets[i](ob).detach()).view(-1).item()))
        return actions

    def train(self):
        # In this function, we will train our network.
        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        state = self.env.reset()

        while True:

            action = self.greedy_policy_Q(state)
            next_state,reward,done,_ = self.env.step(action)
            
            # next_state = torch.tensor(next_state).view(1,-1).float()
            # actions = torch.tensor(actions).view(1).long()
            # reward = torch.tensor(reward).view(1).float()
            transition = [torch.tensor(state).view(1,-1).float(),torch.tensor(action).view(1,-1).long(),\
                            torch.tensor(reward).view(1).float(),torch.tensor(next_state).view(1,-1).float(),done]


            self.replay_memory.append(transition)
            mini_batch = self.replay_memory.sample_batch(self.batch_size) #list

            losses = []
            # calculate each network's loss
            for i in range(self.env.n):
                j = (i+1)%self.env.n

                # here is the training code
                y = torch.tensor([]).float()
                v_actions = torch.tensor([]).long()
                actions = torch.tensor([]).long()
                u_states = torch.tensor([]).float()
                u_states_next = torch.tensor([]).float()

                for transition in mini_batch:
                    sample_state = transition[0]
                    sample_action_index = transition[1]

                    sample_action = self.direction[transition[1]]
                    sample_reward = transition[2]
                    sample_next_state = transition[3]
                    sample_done = transition[-1]

                    u_state_next = torch.cat([sample_next_state,sample_action[:,:j]],dim = 1) 
                    u_states_next = torch.cat((u_states_next,u_state_next))

                    u_state = torch.cat([sample_state,sample_action[:,:i]],dim=1)
                    u_states = torch.cat([u_states,u_state])

                    v_action = sample_action[:,i]
                    v_actions = torch.cat([v_actions,v_action])

                    action = sample_action_index[:,i]
                    actions = torch.cat([actions,action])

                    if sample_done:
                        y = torch.cat((y,sample_reward))
                    else:
                        y = torch.cat([y,sample_reward + self.gamma * torch.max(self.agent.target_nets[j](u_states_next).detach())])
                
                q = torch.gather(self.agent.evaluate_nets[i](u_states),1,actions.view(-1,1))
                losses.append(F.mse_loss(y.view(-1,1),q))
            
            for i in range(self.env.n):
                self.agent.optimizers[i].zero_grad()
                losses[i].backward()
                self.agent.optimizers[i].step()

            self.c += 1
            if self.c % self.update == 0:
                path = self.agent.save_model_weights()
                self.agent.load_model(path)
            if not done:
                state = next_state
            else:
                break

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating average cummulative rewards (returns) for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        pass
        with torch.no_grad():
            test_trial = 20
            # self.current_network.model.eval()

            total_rewards = []
            for _ in range(test_trial):
                state = self.env.reset()
                rewards = 0
                while True:
                    if self.render:
                        self.env.render()

                    action = self.epsilon_greedy_policy(state,train=False)
                    next_state, reward, done, _ = self.env.step(action)
                    rewards += reward
                    state = next_state
                    if done:
                        break
                total_rewards.append(rewards)

            total_rewards = np.array(total_rewards)
            reward_mean = np.mean(total_rewards)
            reward_std = np.sqrt(np.mean(np.sum((total_rewards - reward_mean)**2)/test_trial))   

            # self.current_network.model.train()
        return reward_mean, reward_std

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        pass
        num = 0
        current_state = self.env.reset()
        
        while num < self.replay_memory.burn_in:
                
            action = np.random.choice(self.env.action_space[0],self.env.n)
            next_state, reward, done, _ = self.env.step(action)

            # the structure of transition:
            # [current state, the type of cube, the position of cube, the reward, and the next state]
            transition = [torch.tensor(current_state).view(1,-1).float(),torch.tensor(action).view(1,-1).long(),\
                            torch.tensor(reward).view(1).float(),torch.tensor(next_state).view(1,-1).float(),done]

            self.replay_memory.append(transition)
            num += 1
            if done:
                current_state = self.env.reset()
            if not done:
                current_state = next_state

    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Multi Action Network Argument Parser')
    parser.add_argument('--env',dest='env',type=int,default="4")
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    num_dimension = args.env
    lr = args.lr
    render = args.render
    num_trails = 10
    num_episodes = 1000
    SAVE_STR = 'MAN-' + str(num_dimension) + 'd'

    reward_means_total = []
    reward_stds_total = []

    for trail in tqdm.tqdm(range(num_trails)):
        reward_means = []
        reward_stds = []
        agent = MAN_Agent(num_dimension,lr,render=render)
        for epi in range(num_episodes):
            if epi % 10 == 0:
                reward_mean, reward_std = agent.test()
                reward_means.append(reward_mean)
                reward_stds.append(reward_std)
                print("The test reward for episode %d is %.1f."%(epi, reward_means[-1]))
                print('The epsilon is:',agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * math.exp(-1. * agent.c / agent.epsilon_decay))

                # save the best model
                if reward_means[-1] == max(reward_means) or epi% 100 == 0:
                    path = agent.agent.save_best_model()
                    print("The best model is saved with reward %.1f."%(reward_means[-1]))
            agent.train()

        reward_means_total.append(reward_means)
        reward_stds_total.append(reward_stds)

    # save the data in a csv file for plotting later on
    pd.DataFrame(np.array(reward_means_total).mean(axis=0)).to_csv('./data/' + SAVE_STR + '_mean.csv', header=None, index=None)
    pd.DataFrame(np.array(reward_stds_total).mean(axis=0)).to_csv('./data/' + SAVE_STR + '_std.csv', header=None, index=None)
    
    plt.plot(np.array(reward_means_total).mean(axis=0))
    # plt.show()
    plt.savefig("./plots/reward_"+SAVE_STR)

if __name__ == '__main__':
    main(sys.argv)


