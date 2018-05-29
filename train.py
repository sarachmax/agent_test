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
all_index = end_index-start_index
for i in range(state_size, all_index):
    X_train.append(train_data[i-state_size:i,0])
X_train = np.array(X_train)


class TrainEnvironment:
    def __init__(self, data, num_index):
        self.train_data = data
        self.train_index = 0 
        self.end_index = num_index
        self.loss_limit = 0.0005 # force sell 
        self.profit = 0.0 
        self.reward = 0
        
    def reset(self):
        self.train_index = 0 
        self.profit = 0
        self.reward = 0 
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
        self.reward = 0
        action = self.get_action(action)
        self.profit += action*price_diff
        
        if price_diff*action > 0 :
            self.reward = price_diff
        elif price_diff*action < 0 : 
            self.reward = 3*price_diff*action
        elif price_diff == 0 and action == 0 : 
            self.reward = 0.002
        elif action == 0 :
            self.reward = -0.001
    
    def done_check(self):
        loss = -self.loss_limit*self.train_data[self.train_index,59:60]
        print('loss limit : ', loss)
        if self.train_index + 1 == self.end_index :
            if self.profit > 0 : 
                self.reward = self.profit 
            return True 
        elif self.profit <= loss : 
            self.reward = -1
            return True
        else :
            return False
        
    def step(self,action):
        state_size = 60
        skip = random.randrange(state_size/2,state_size-1)
        print('skip index : ', skip)
        self.train_index += skip
        if self.train_index >= self.end_index-1 : 
            self.train_index = self.end_index-1 
        ns = [X_train[self.train_index]]
        self.calculate_reward(action)
        done = self.done_check()
        return ns, self.reward, done

#########################################################################################################
# Train     
#########################################################################################################         
def watch_result(episode ,s_time, e_time, c_index, all_index, action, reward, profit):
    print('-------------------- Check -------------------------')
    print('start time: ' + s_time)  
    print('counter : ', c_index,'/', all_index,' of episode : ', episode)
    print('action : ', action)
    print('reward : ', reward)
    print('current profit : ', profit)
    print('end_time: ' + e_time)
    print('-------------------End Check -----------------------')

    
if __name__ == "__main__":
    
    agent = DQNAgent(state_size)
    #agent.load("agent_model.h5")
    num_index = all_index - state_size
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
                print('----------------------------- Episode Result -----------------------')
                print("episode: {}/{}, time: {}, e: {:.2}"
                      .format(e, EPISODES, t, agent.epsilon))
                print('----------------------------- End Episode --------------------------')
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            end_time = str(datetime.datetime.now().time())
            
            watch_result(e , start_time, end_time, env.train_index, end_index-start_index, env.get_action(action), reward ,env.profit)     
                     
    agent.save("agent_model.h5")
                      
    