# this file is used to visualize the rewards from different models in one figure.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d2_means = pd.read_csv('./data/DQN-2d_mean.csv',header=None).to_numpy()
d2_stds = pd.read_csv('./data/DQN-2d_std.csv',header=None).to_numpy()

d3_means = pd.read_csv('./data/DQN-3d_mean.csv',header=None).to_numpy()
d3_stds = pd.read_csv('./data/DQN-3d_std.csv',header=None).to_numpy()

d4_means = pd.read_csv('./data/DQN-4d_mean.csv',header=None).to_numpy()
d4_stds = pd.read_csv('./data/DQN-4d_std.csv',header=None).to_numpy()

d5_means = pd.read_csv('./data/DQN-5d_mean.csv',header=None).to_numpy()
d5_stds = pd.read_csv('./data/DQN-5d_std.csv',header=None).to_numpy()

d6_means = pd.read_csv('./data/DQN-6d_mean.csv',header=None).to_numpy()
d6_stds = pd.read_csv('./data/DQN-6d_std.csv',header=None).to_numpy()

MAN_d2_means = pd.read_csv('./data/MAN-2d_mean.csv',header=None).to_numpy()
MAN_d2_stds = pd.read_csv('./data/MAN-2d_std.csv',header=None).to_numpy()

MAN_d3_means = pd.read_csv('./data/MAN-3d_mean.csv',header=None).to_numpy()
MAN_d3_stds = pd.read_csv('./data/MAN-3d_std.csv',header=None).to_numpy()

MAN_d4_means = pd.read_csv('./data/MAN-4d_mean.csv',header=None).to_numpy()
MAN_d4_stds = pd.read_csv('./data/MAN-4d_std.csv',header=None).to_numpy()

MAN_d5_means = pd.read_csv('./data/MAN-5d_mean.csv',header=None).to_numpy()
MAN_d5_stds = pd.read_csv('./data/MAN-5d_std.csv',header=None).to_numpy()

MAN_d6_means = pd.read_csv('./data/MAN-6d_mean.csv',header=None).to_numpy()
MAN_d6_stds = pd.read_csv('./data/MAN-6d_std.csv',header=None).to_numpy()

plt.figure(figsize=(8,8),dpi=300)
plt.xlabel("Episode")
plt.ylabel("reward")
plt.ylim(-20,100)
plt.title('MAM reward curve in n-dimension maze task')

# plt.plot(np.arange(len(d2_means)), d2_means,color = 'b',alpha = 0.8,label='DQN 2D')
# plt.plot(np.arange(len(d3_means)), d3_means,color = 'c',alpha = 0.8,label='DQN 3D')
# plt.plot(np.arange(len(d4_means)), d4_means,color = 'r',alpha = 0.8,label='DQN 4D')
# plt.plot(np.arange(len(d5_means)), d5_means,color = 'g',alpha = 0.8,label='DQN 5D')
# plt.plot(np.arange(len(d6_means)), d6_means,color = 'm',alpha = 0.8,label='DQN 6D')

plt.plot(np.arange(len(MAN_d2_means)), MAN_d2_means,color = 'b',alpha = 0.8,label='MAN 2D')
plt.plot(np.arange(len(MAN_d3_means)), MAN_d3_means,color = 'c',alpha = 0.8,label='MAN 3D')
plt.plot(np.arange(len(MAN_d4_means)), MAN_d4_means,color = 'r',alpha = 0.8,label='MAN 4D')
plt.plot(np.arange(len(MAN_d5_means)), MAN_d5_means,color = 'g',alpha = 0.8,label='MAN 5D')
plt.plot(np.arange(len(MAN_d6_means)), MAN_d6_means,color = 'm',alpha = 0.8,label='MAN 6D')

# plt.fill_between(np.arange(len(d2_means)), d2_means[:,0]+d2_stds[:,0], d2_means[:,0]-d2_stds[:,0],alpha= 0.1,color = 'b',linewidth=0.0)
# plt.fill_between(np.arange(len(d3_means)), d3_means[:,0]+d3_stds[:,0], d3_means[:,0]-d3_stds[:,0],alpha= 0.1,color = 'c',linewidth=0.0)
# plt.fill_between(np.arange(len(d4_means)), d4_means[:,0]+d4_stds[:,0], d4_means[:,0]-d4_stds[:,0],alpha= 0.1,color = 'r',linewidth=0.0)
# plt.fill_between(np.arange(len(d5_means)), d5_means[:,0]+d5_stds[:,0], d5_means[:,0]-d5_stds[:,0],alpha= 0.1,color = 'g',linewidth=0.0)
# plt.fill_between(np.arange(len(d6_means)), d6_means[:,0]+d6_stds[:,0], d6_means[:,0]-d6_stds[:,0],alpha= 0.1,color = 'm',linewidth=0.0)

plt.fill_between(np.arange(len(MAN_d2_means)), MAN_d2_means[:,0]+MAN_d2_stds[:,0], MAN_d2_means[:,0]-MAN_d2_stds[:,0],alpha= 0.1,color = 'b',linewidth=0.0)
plt.fill_between(np.arange(len(MAN_d3_means)), MAN_d3_means[:,0]+MAN_d3_stds[:,0], MAN_d3_means[:,0]-MAN_d3_stds[:,0],alpha= 0.1,color = 'c',linewidth=0.0)
plt.fill_between(np.arange(len(MAN_d4_means)), MAN_d4_means[:,0]+MAN_d4_stds[:,0], MAN_d4_means[:,0]-MAN_d4_stds[:,0],alpha= 0.1,color = 'r',linewidth=0.0)
plt.fill_between(np.arange(len(MAN_d5_means)), MAN_d5_means[:,0]+MAN_d5_stds[:,0], MAN_d5_means[:,0]-MAN_d5_stds[:,0],alpha= 0.1,color = 'g',linewidth=0.0)
plt.fill_between(np.arange(len(MAN_d6_means)), MAN_d6_means[:,0]+MAN_d6_stds[:,0], MAN_d6_means[:,0]-MAN_d6_stds[:,0],alpha= 0.1,color = 'm',linewidth=0.0)

plt.legend(loc='upper left')
# plt.show()
plt.grid(True,color='w',linestyle='-',linewidth=0.5)
plt.gca().patch.set_facecolor('0.9')
plt.savefig('./plots/compare_MAN')