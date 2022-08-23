from datetime import datetime
import re
import os
import torch
import matplotlib.pyplot as plt

def date_in_string():
    time = str(datetime.now())
    time = re.sub(' ', '_', time)
    time = re.sub(':', '', time)
    time = re.sub('-', '_', time)
    time = time[0:15]
    return time

time = date_in_string()

def save_log_loss(i_steps, loss_I, loss_II):
    path = os.getcwd()+ r'\log'
    with open(path + r'\log_loss_'+ time + '.txt', 'a') as outfile:
        outfile.write(date_in_string() + '\t' + str(i_steps) + '\t' + str(loss_I) + '\t' + str(loss_II) + '\n')
    return

def save_log_score(i_episodes,mean_r,max_r):
    path = os.getcwd()+ r'\log'
    with open(path + r'\log_score_' + time + '.txt', 'a') as outfile:
        outfile.write(date_in_string() + '\t' + str(i_episodes) + '\t' + str(mean_r) + '\t' + str(max_r) +'\n')
    return

def save_eva_score(e_rewards,num_episode,step):
    path = os.getcwd()+ r'\result'
    with open(path + r'\score_' + time + '.txt', 'a') as outfile:

        outfile.write("%.2f, %d, %d\n" % (float(sum(e_rewards))/float(num_episode), step, num_episode))
    return

def save_model_params(model1,model2):
    path = r'.\model'
    torch.save(model1.state_dict(), path + r'\MAN_I_' + time + '.pkl')
    torch.save(model2.state_dict(), path + r'\MAN_II_' + time + '.pkl')
    return

def save_qscore_figure():
    path = os.getcwd()+ r'\log'
    with open(path + r'\log_loss_'+ time + '.txt') as f:
        lines = f.readlines()
    avg_qscore = []
    for line in lines:
        avg_qscore.append(float(line.split()[3]))
    path = os.getcwd()+ r'\figure'
    plt.plot(avg_qscore)
    plt.savefig(path + r'\avg_qscore_' + time + '.png')
    plt.clf()

def save_score_figure(env_name):
    path = os.getcwd()+ r'\result'
    with open(path + '\score_' + time +'.txt') as f:
        lines = f.readlines()
    avg_score = []
    for line in lines:
        avg_score.append(float(line.split(',')[0]))
    path = os.getcwd()+ r'\figure'
    x = range(0,int(5e4*len(avg_score)),int(5e4))
    plt.plot(x, avg_score)
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward per Episodes')
    plt.title('Average Reward on ' + env_name)
    plt.savefig(path + r'\avg_score_' + time + '.png')
    plt.clf()