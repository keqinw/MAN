from man import MAN
import utilities as u
import torch
import matplotlib.pyplot as plt
from atari_wrappers import wrap_deepmind, make_atari
import argparse
LEARN_START = 50000#50000

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="FrostbiteNoFrameskip-v4",
                    help='name of environement')
args = parser.parse_args()
myDQQN = MAN(args.env_name)
NUM_STEPS =10000000 # 5000000
TARGET_NET_UPDATE_FREQUENCY = 10000 #10000
NUM_EVA = 20
max_r = 0
mean_r= 0
loss_I=0
loss_II=0
avg_q =0
running_loss_I = 0
running_loss_II = 0
max_mean_r = 0
sum_r = 0
done =True
e_rewards_max = -1e6
i = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

progressive = tqdm(range(NUM_STEPS), total=NUM_STEPS, ncols=100, leave=False, unit='steps')

for step in progressive:

    if done and myDQQN.env.was_real_done == True:
        mean_r += sum_r
        if sum_r >max_r :
            max_r = sum_r

        if (i+1)%20 == 0:
            u.save_log_score(i,mean_r/20, max_r)
            print('    Trained {} games, with mean reward {}, max reward {}'.format(i+1,mean_r/20,max_r))
            if mean_r>max_mean_r:
                max_mean_r = mean_r
                u.save_model_params(myDQQN.eva_net_I,myDQQN.eva_net_II)
            max_r,mean_r=0,0

        sum_r = 0
        i +=1

    if done :
        s = myDQQN.state_initialize()
        # img,_,_,_ = myDQQN.env.step(1)
 
    a_I = myDQQN.choose_action_I(s)
    a_i = torch.tensor(a_I,device=device,dtype=torch.int64).to(device)
    a_II = myDQQN.choose_action_II(s,a_i)
    a = myDQQN.choose_action(a_I,a_II)
        
    img,r,done,info = myDQQN.env.step(a)
    sum_r += r
    myDQQN.state_buffer.pop(0)
    myDQQN.state_buffer.append(img)


    s_ = myDQQN.state_buffer[1:5]
    myDQQN.store_transition(s,a_I,a_II,r,s_,done)
    s = s_

    if len(myDQQN.memory)>LEARN_START and myDQQN.state_counter%4 :
        loss_I,loss_II = myDQQN.learn()
        running_loss_I += loss_I
        running_loss_II += loss_II
    # if myDQQN.state_counter%500 ==0:
    #     running_loss /= 250
    if myDQQN.state_counter % TARGET_NET_UPDATE_FREQUENCY ==0:
        running_loss_I /= TARGET_NET_UPDATE_FREQUENCY
        running_loss_II /= TARGET_NET_UPDATE_FREQUENCY
        u.save_log_loss(step,running_loss_I,running_loss_II)
        myDQQN.target_net_I.load_state_dict(myDQQN.eva_net_I.state_dict())
        myDQQN.target_net_II.load_state_dict(myDQQN.eva_net_II.state_dict())
        running_loss = 0
    # if myDQQN.state_counter%500 ==0:
    #     running_loss = 0
    if (myDQQN.state_counter+1) %50000 ==0 :
        env = make_atari(args.env_name)
        env = wrap_deepmind(env, frame_stack=False, episode_life=False, clip_rewards=False)
        e_rewards = myDQQN.evaluate(myDQQN.env_raw,  num_episode=NUM_EVA)
        u.save_eva_score(e_rewards,NUM_EVA,step+1)
        if sum(e_rewards)/NUM_EVA >= e_rewards_max:
            u.save_model_params(myDQQN.eva_net_I,myDQQN.eva_net_II)
            e_rewards_max = sum(e_rewards)/NUM_EVA
        


        u.save_qscore_figure()
        u.save_score_figure('Frostbite')