import random
import gym
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import losses

###############################
#
# Create Lunar Lander-V2 enviroment
#
###############################

seed = 2031
env = gym.make('LunarLander-v2')
np.random.seed(seed)
random.seed(seed)
env.seed(seed)

# Define Neural Network Q-Learning Agent

class NNQ_Agent:
    def __init__(self, state_dim, action_dim, gamma, epsilon, learning_rate, loss_fuction):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = gamma    
        self.epsilon = epsilon  
        self.learning_rate = learning_rate
        self.predict_model = self._build_model(loss_fuction)
        self.target_model = self._build_model(loss_fuction)
        self.learning_mode = False # Indicator: if the agent starts learning
        self.step = 0 # record of running steps
        self.minibatch_size = 32 # size of mini batch for training
    
    def _build_model(self, loss_fuction):
        model = Sequential()
        model.add(Dense(40, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        
        if (loss_fuction == "logcosh"):
        	print("Using logcosh!")
        	model.compile(loss=losses.logcosh, optimizer=Adam(lr=self.learning_rate), metrics=['acc'])
        else:
        	print("Using mse!")
        	model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self.learning_rate), metrics=['acc'])
        return model

    def _memory(self, state, action, reward, next_state, done):
        
        # record (s, a, s', r, done) for each observation
        self.memory.append((state, action, reward, next_state, done))
        
        # update target model for every 500 steps
        if self.step % 500 == 0:
            self.target_model.set_weights(self.predict_model.get_weights())
            
        # step count ++
        self.step = self.step + 1

    def _take_action(self, state, isTraining):
        
        # Do not carry out epsilon greedy policy for test trails
        if isTraining:
            if (np.random.uniform(0,1) < self.epsilon) | (not self.learning_mode):
                return env.action_space.sample()
        state = np.reshape(state, [1, self.state_dim])
        action = self.predict_model.predict(state) # an action-reward matrix
        return np.argmax(action[0])  # returns action with maximum reward

    def _Q_learning(self):
        
        if len(self.memory) < 1000: # start batch learning while we have more than 1000 samples in memory
            return([0.0, 0.0]) # loss = 0.0, acc = 0.0
        if not self.learning_mode:
            self.learning_mode = True # star learning mode
        
        # Initializing input array and output array
        X = np.zeros((self.minibatch_size, self.state_dim)) # 32 x 8 matrix
        Y = np.zeros((self.minibatch_size, self.action_dim)) # 32 x 4 matrix
        
        # random sampling memory
        minibatch = random.sample(self.memory, self.minibatch_size)
        index = 0
        for state, action, reward, next_state, done in minibatch:
            
            state = np.reshape(state, [1, self.state_dim])
            next_state = np.reshape(next_state, [1, self.state_dim])
            
            # Get action-reward relation matrix for current state: using predict_model
            Q_current = self.predict_model.predict(state)
            
            # Get action-reward relation matrix for current state: using target_model
            Q_next = self.target_model.predict(next_state)
            
            current_action_reward = reward # True if episode is done.
            # If episode is not done, there is a need for computing future reward with discount gamma
            if (not done):
                current_action_reward = current_action_reward + self.gamma * np.amax(Q_next[0])
            
            # Update action-reward relation matrix for current state
            Q_current[0][action] = current_action_reward
            
            # Assign input array and output array
            X[index,:] = state[0]
            Y[index,:] = Q_current[0]
            
            index = index + 1
               
        # Using X and Y for one step learning with predict_model
        results = self.predict_model.fit(X, Y, epochs=1, batch_size=self.minibatch_size, verbose=False)
        
        # Evaluate the fitting model:
        scores = self.predict_model.evaluate(X, Y, verbose=False)
        
        return(scores)

    def _load_model(self, name, model_type):
        if model_type == "predict_model":
            self.predict_model.load_weights(name)
        if model_type == "target_model":
            self.target_model.load_weights(name)

    def _save_model(self, name, model_type):
        if model_type == "predict_model":
            self.predict_model.save_weights(name)
        if model_type == "target_model":
            self.target_model.save_weights(name)

###############################
#
# Training Agent Function
#
###############################

def training_NNQ_Agent(EPISODES, gamma, epsilon, learning_rate, loss_fuction, h5_name):
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = NNQ_Agent(state_dim, action_dim, gamma, epsilon, learning_rate, loss_fuction)
    
    total_episode_reward = []
    total_episode_loss = []
    total_episode_acc = []
    total_episode_step = []
    total_epsilon = []

    for e in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        episode_acc = 0
        episode_step = 0
        
        total_epsilon.append(agent.epsilon)
        
        while (True):
            
            # Taking an action with epsilon greedy policy
            action = agent._take_action(state, True)
            next_state, reward, done, _ = env.step(action)
            
            # Save memory
            agent._memory(state, action, reward, next_state, done)
            
            # Carry out Q learning with NN
            scores = agent._Q_learning()
            #print("scores: ", scores)
            
            # Update state
            state = next_state
            
            # Record results
            episode_step += 1
            episode_reward += reward
            episode_loss += scores[0]
            episode_acc += scores[1]

            if episode_step > 5000:
                print("episode: {}/{}, score: {}, total: {}, time: {}, done: {}".format(e, EPISODES, reward, episode_reward, episode_step, done))
                break
            
            if reward == 100:
                print("episode: {}/{}, score: {}, total: {}, time: {}, loss: {:.2}, done: {}, agent.epsilon: {}".format(e, EPISODES, reward, episode_reward, episode_step, episode_loss/float(episode_step), done, agent.epsilon))
                break
            
            if reward == -100:
                print("episode: {}/{}, score: {}, total: {}, time: {}, loss: {:.2}, done: {}, agent.epsilon: {}".format(e, EPISODES, reward, episode_reward, episode_step, episode_loss/float(episode_step), done, agent.epsilon))
                break
        
        # decaying epsilon when the episode is done: non-linear decay
        if agent.epsilon * 0.995 > 0.01:
            agent.epsilon = agent.epsilon * 0.995
                        
        # Save info for each episode
        total_episode_reward.append(episode_reward)
        total_episode_loss.append(episode_loss)
        total_episode_acc.append(episode_acc)
        total_episode_step.append(episode_step)
        
    
    # Save weighting factors:
    agent._save_model("./LL_predict_model-dqn" + h5_name, "predict_model")
    agent._save_model("./LL_target_model-dqn" + h5_name, "target_model")
    
    # Return the training results:
    return[total_episode_reward, total_episode_loss, total_episode_acc, total_episode_step, total_epsilon, agent.predict_model]

###############################
#
# Testing Agent Fucntion
#
###############################

def testing_NNQ_Agent(EPISODES, gamma, epsilon, learning_rate, loss_fuction, model):
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = NNQ_Agent(state_dim, action_dim, gamma, epsilon, learning_rate, loss_fuction)
    agent.predict_model = model
    
    total_episode_reward = []
    total_episode_step = []

    for e in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        episode_step = 0
        
        while (True):
            
            # Taking an action without epsilon greedy policy
            action = agent._take_action(state, False)
            next_state, reward, done, _ = env.step(action)
            
            # Update state
            state = next_state
            
            # Record results
            episode_step += 1
            episode_reward += reward

            if episode_step > 5000:
                print("episode: {}/{}, score: {}, total: {}, time: {}, done: {}".format(e, EPISODES, reward, episode_reward, episode_step, done))
                break
            
            if reward == 100:
                print("episode: {}/{}, score: {}, total: {}, time: {}, done: {}".format(e, EPISODES, reward, episode_reward, episode_step, done))
                break
            
            if reward == -100:
                print("episode: {}/{}, score: {}, total: {}, time: {}, done: {}".format(e, EPISODES, reward, episode_reward, episode_step, done))
                break
            
        # Save info for each episode
        total_episode_reward.append(episode_reward)
        total_episode_step.append(episode_step)
    
    # Return the training results:
    return[total_episode_reward, total_episode_step]

###############################
#
# Using mse as loss function, gamma = 0.99, epsilon = 1.0, learning_rate = 0.001, 1000 EPISODES
#
###############################

discount = 0.99

training_results_mse = training_NNQ_Agent(1000, discount, 1.0, 0.001, "mse", "_Gamma_" + str(discount) + "_mse.h5")
training_results_mse_array = np.array(training_results_mse[:5])
training_results_mse_out = pd.DataFrame(training_results_mse_array.T)
training_results_mse_out.columns = ["Total_Reward", "Total_Loss", "Total_Accuracy", "Steps", "Epsilons"]
training_results_mse_out.to_csv("Gamma_" + str(discount) + "_training_results_mse_out.csv", index=False)

###############################
#
# Using mse as loss function, gamma = 0.99, epsilon = 1.0, learning_rate = 0.001, 100 trails
#
###############################

testing_results_mse = testing_NNQ_Agent(100, discount, 1.0, 0.001, "mse", training_results_mse[5])

testing_results_mse_array = np.array(testing_results_mse)
testing_results_mse_out = pd.DataFrame(testing_results_mse_array.T)
testing_results_mse_out.columns = ["Total_Reward", "Steps"]
testing_results_mse_out.to_csv("Gamma_" + str(discount) + "_testing_results_mse_out.csv", index=False)






