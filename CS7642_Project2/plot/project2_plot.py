
# coding: utf-8

# In[36]:

from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# In[7]:

def movingAverage(inputList):
    outputList = []
    for i in range(len(inputList)):
        end = i + 50
        windows = []
        if end < len(inputList):
            windows = inputList[i:end]
        else:
            windows = inputList[i:len(inputList)]
        outputList.append(np.mean(windows))
    return(outputList)


# In[37]:

os.chdir(sys.argv[1])
#print(sys.argv[0])


# In[15]:

prefix = ["Gamma_", "Alpha_"]
suffix = ["_training_results_mse_out.csv", "_testing_results_mse_out.csv"]


# In[24]:

figureList = []

for file_suffix in suffix:
    for gamma in [0.2, 0.4, 0.6, 0.8, 0.99]:
        print(gamma)
        df = pd.read_csv(prefix[0] + str(gamma) + file_suffix)
        MA = movingAverage(list(df["Total_Reward"]))
        figureList.append(MA)


# In[31]:

plt.figure(figsize=(10,10))

plt.plot(range(1000), figureList[0], 'o-')
plt.plot(range(1000), figureList[1], 'o-')
plt.plot(range(1000), figureList[2], 'o-')
plt.plot(range(1000), figureList[3], 'o-')
plt.plot(range(1000), figureList[4], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Episodes')
plt.title('Figure1: Tuning Up Gamma: 1000 episodes')

plt.legend(['Gamma=0.2', 'Gamma=0.4', 'Gamma=0.6', 'Gamma=0.8', 'Gamma=0.99'], loc='upper left')
plt.savefig('tuning.up.gamma.png')


# In[32]:

plt.figure(figsize=(10,10))

plt.plot(range(100), figureList[5], 'o-')
plt.plot(range(100), figureList[6], 'o-')
plt.plot(range(100), figureList[7], 'o-')
plt.plot(range(100), figureList[8], 'o-')
plt.plot(range(100), figureList[9], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Trails')
plt.title('Figure2: Different Gamma: 100 trails')

plt.legend(['Gamma=0.2', 'Gamma=0.4', 'Gamma=0.6', 'Gamma=0.8', 'Gamma=0.99'], loc='upper left')
plt.savefig('trails.gamma.png')


# In[33]:

figureList = []

for file_suffix in suffix:
    for alpha in [0.0001, 0.001, 0.01, 0.1, 0.99]:
        df = pd.read_csv(prefix[1] + str(alpha) + file_suffix)
        MA = movingAverage(list(df["Total_Reward"]))
        figureList.append(MA)


# In[34]:

plt.figure(figsize=(10,10))

plt.plot(range(1000), figureList[0], 'o-')
plt.plot(range(1000), figureList[1], 'o-')
plt.plot(range(1000), figureList[2], 'o-')
plt.plot(range(1000), figureList[3], 'o-')
plt.plot(range(1000), figureList[4], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Episodes')
plt.title('Figure3: Tuning Up Alpha: 1000 episodes')

plt.legend(['Alpha=0.0001', 'Alpha=0.001', 'Alpha=0.01', 'Alpha=0.1', 'Alpha=0.99'], loc='upper left')
plt.savefig('tuning.up.alpha.png')


# In[35]:

plt.figure(figsize=(10,10))

plt.plot(range(100), figureList[5], 'o-')
plt.plot(range(100), figureList[6], 'o-')
plt.plot(range(100), figureList[7], 'o-')
plt.plot(range(100), figureList[8], 'o-')
plt.plot(range(100), figureList[9], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Trails')
plt.title('Figure4: Different Alpha: 100 trails')

plt.legend(['Alpha=0.0001', 'Alpha=0.001', 'Alpha=0.01', 'Alpha=0.1', 'Alpha=0.99'], loc='upper left')
plt.savefig('trails.alpha.png')


# In[38]:

figureList = []
df = pd.read_csv("Alpha_0.001_training_results_mse_out.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)
df = pd.read_csv("logcosh_training_results_mse_out.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)
df = pd.read_csv("Alpha_0.001_testing_results_mse_out.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)
df = pd.read_csv("logcosh_testing_results_mse_out.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)


# In[39]:

plt.figure(figsize=(10,10))

plt.plot(range(1000), figureList[0], 'o-')
plt.plot(range(1000), figureList[1], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Episodes')
plt.title('Figure5: loss function: 1000 episodes')

plt.legend(['MSE', 'LOGCOSH'], loc='upper left')
plt.savefig('loss.training.png')


# In[41]:

plt.figure(figsize=(10,10))

plt.plot(range(100), figureList[2], 'o-')
plt.plot(range(100), figureList[3], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Trails')
plt.title('Figure6: loss function: 100 trails')

plt.legend(['MSE', 'LOGCOSH'], loc='upper left')
plt.savefig('loss.trails.png')


# In[42]:

figureList = []
df = pd.read_csv("Alpha_0.001_training_results_mse_out.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)
eps = list(df["Epsilons"])
figureList.append(eps)
df = pd.read_csv("linearE_training_results_mse_out.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)
eps = list(df["Epsilons"])
figureList.append(eps)


# In[43]:

plt.figure(figsize=(10,10))

plt.plot(figureList[1], figureList[0], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Epsilons')
plt.title('Figure7: non-linear epsilons-decay: 1000 episodes')
plt.savefig('non-LinearE.png')


# In[44]:

plt.figure(figsize=(10,10))

plt.plot(figureList[3], figureList[2], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Epsilons')
plt.title('Figure8: linear epsilons-decay: 1000 episodes')
plt.savefig('LinearE.png')


# In[45]:

# Final


# In[46]:

figureList = []
df = pd.read_csv("final_model_training_results_mse_out.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)
df = pd.read_csv("final_model_testing_results_mse_out.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)


# In[49]:

plt.figure(figsize=(10,10))

plt.plot(range(1500), figureList[0], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Episodes')
plt.title('Figure9: final model training: 1500 episodes')
plt.savefig('final_model_training.png')


# In[50]:

plt.figure(figsize=(10,10))

plt.plot(range(100), figureList[1], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Trails')
plt.title('Figure10: final model testing: 100 trails')
plt.savefig('final_model_testing.png')


# In[51]:

figureList = []
df = pd.read_csv("final_model_testing_results_mse_out.less5000.csv")
MA = movingAverage(list(df["Total_Reward"]))
figureList.append(MA)


# In[52]:

plt.figure(figsize=(10,10))

plt.plot(range(91), figureList[0], 'o-')

plt.ylabel('Rolling Average Rewards (50 window)')
plt.xlabel('Trails')
plt.title('Figure11: final model testing: 91 trails')
plt.savefig('final_model_testing.91trails.png')


# In[ ]:



