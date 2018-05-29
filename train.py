# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:11:15 2018

@author: sarac
"""

from EURUSDagent import DQNAgent
import datetime
import random 
import numpy as np
import pandas as pd 

EPISODES = 100

start_index = 45    #2010.01.01 00:00
end_index = 3161+1  #2012.12.30 20:00
dataset = pd.read_csv('EURUSD_4H.csv')
train_data = dataset.iloc[start_index:end_index,5:6]

train_data = np.array(train_data)
state_size = 60
X_train = [] 
for i in range(state_size, end_index-start_index):
    X_train.append(train_data[i-state_size:i,0])
X_train = np.array(X_train)


class TrainEnvironment:
    def __init__(self, data, num_index):
        self.train_data = data
        self.train_index = 0 
        self.end_index = num_index
        self.loss_limit = 0.008 
        self.profit = 0.0 
        
    def reset(self):
        self.train_index = 0 
        self.profit = 0
        return [self.train_data[self.train_index]]
    
    def get_action(self,action):
        if action == 0 :
            # buy 
            return 1
        elif action == 1 : 
            # sell 
            return -1
        else : 
            # noaction 
            return 0 
    
    def calculate_reward(self, action):
        price_diff = self.train_data[self.train_index,59:60] - self.train_data[self.train_index,58:59]
        self.profit += price_diff
        reward = 0
        action = self.get_action(action)
        if price_diff*action > 0 :
            reward = price_diff
        elif price_diff*action < 0 : 
            reward = 3*price_diff*action
        elif price_diff == 0 and action == 0 : 
            reward = 0.002
        elif action == 0 :
            reward = -0.001
        return reward 
    
    def done_check(self):
        loss = -self.loss_limit*self.train_data[self.train_index,59:60]
        print('loss limit : ', loss)
        if self.train_index + 1 == self.end_index :
            return True 
        else :
            return self.profit <= loss
        
    def step(self,action):
        state_size = 60
        skip = random.randrange(1,state_size/2)
        print('skip index : ', skip)
        self.train_index += skip
        if self.train_index >= self.end_index-1 : 
            self.train_index = self.end_index-1 
        done = self.done_check()
        ns = [X_train[self.train_index]]
        reward = self.calculate_reward(action)
        return ns, reward, done
        
    
if __name__ == "__main__":
    
    agent = DQNAgent(state_size)
    #agent.load("agent_model.h5")
    num_index = end_index - start_index
    env = TrainEnvironment(X_train, num_index)
    batch_size = 32 
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, (1, state_size, 1))
        
        for t in range(end_index-start_index):
            start_time = str(datetime.datetime.now().time())
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, (1,state_size,1))
            agent.remember(state, action, reward, next_state, done)
            state = next_state       
            if done:
                agent.update_target_model()
                print('------------------- Episode result -----------------------')
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, t, agent.epsilon))
                print('-------------------End Episode -----------------------')
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            end_time = str(datetime.datetime.now().time())
              
            print('-------------------- Check -------------------------')
            print('start time: ' + start_time)  
            print('counter : ', env.train_index,'/', end_index-start_index)
            print('action : ', env.get_action(action))
            print('reward : ', reward)
            print('current profit : ', env.profit)
            print('end_time: ' + end_time)
            print('-------------------End Check -----------------------')
            
    agent.save("agent_model.h5")
                      
    