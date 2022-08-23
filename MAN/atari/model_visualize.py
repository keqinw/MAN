import gym 
import time
import numpy as np
from man import MAN
import torch
import matplotlib.pyplot as plt
import time
from atari_wrappers import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="FrostbiteNoFrameskip-v4",
                    help='name of environement')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

myDQQN = MAN(args.env_name)
myDQQN.eva_net_I.load_state_dict(torch.load(r'.\model\MAN_I_2022_08_20_1655.pkl'))
myDQQN.eva_net_II.load_state_dict(torch.load(r'.\model\MAN_II_2022_08_20_1655.pkl'))
num_episode = 20
env = wrap_deepmind(myDQQN.env_raw)

e_rewards = []
for i in range(num_episode):
    img = env.reset()
    sum_r = 0
    done = False
    state_buffer=[]
    for i in range(5):
        state_buffer.append(img)
    s = state_buffer[1:5]
    while not done:
        a_I = myDQQN.choose_action_I(s,train=False)
        a_i = torch.tensor(a_I,device=device,dtype=torch.int64).to(device)
        a_II = myDQQN.choose_action_II(s,a_i,train=False)
        a = myDQQN.choose_action(a_I,a_II)
    
        img,r,done,info = env.step(a)
        # render
        # a = env.render('rgb_array')
        # plt.imshow(a)
        # plt.pause(0.01)
        # plt.clf()
        sum_r += r
        state_buffer.pop(0)
        state_buffer.append(img)
        s_ = state_buffer[1:5]
        s = s_

    e_rewards.append(sum_r)

print(np.mean(e_rewards))