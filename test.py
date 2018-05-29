# -*- coding: utf-8 -*-
"""
Created on Tue May 29 17:12:20 2018

@author: sarac
"""

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
import matplotlib.pyplot as plt 

EPISODES = 100

start_index = 45    #2010.01.01 00:00
end_index = 3161+1  #2012.12.30 20:00
dataset = pd.read_csv('EURUSD_4H.csv')
train_data = dataset.iloc[start_index:end_index,5:6]

train_data = np.array(train_data)
state_size = 60
X_test= [] 
all_index = end_index-start_index
for i in range(state_size, all_index):
    X_test.append(train_data[i-state_size:i,0])
X_test = np.array(X_test)


class TestEnvironment:
    def __init__(self, data, num_index):
        self.train_data = data
        self.train_index = 0 
        self.end_index = num_index-1 
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
        price_diff = self.train_data[self.train_index+1,59:60] - self.train_data[self.train_index,58:59]
        self.reward = 0
        action = self.get_action(action)
        self.profit += action*price_diff
        
        if price_diff*action > 0 :
            self.reward = price_diff*action
        elif price_diff*action < 0 : 
            self.reward = 3*price_diff*action
        elif action == 0 :
            self.reward = -0.01
    
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
        skip = 1
        self.train_index += skip
        if self.train_index >= self.end_index-1 : 
            self.train_index = self.end_index-1 
        ns = [self.train_data[self.train_index]]
        self.calculate_reward(action)
        done = self.done_check()
        return ns, self.reward, done

#########################################################################################################
# Train     
#########################################################################################################         
def watch_result(c_index, all_index, action, reward, profit):
    print('-------------------- Check -------------------------')  
    print('counter : ', c_index,'/', all_index)
    print('action : ', action)
    print('reward : ', reward)
    print('current profit : ', profit)
    print('-------------------End Check -----------------------')

    
if __name__ == "__main__":
    
    agent = DQNAgent(state_size)
    agent.load("agent_model.h5")
    num_index = all_index - state_size
    env = TestEnvironment(X_test, num_index)
    
    state = env.reset()
    state = np.reshape(state, (1, state_size, 1))
    profit = []
    batch_size = 32
    
    for t in range(end_index-start_index):
        start_time = str(datetime.datetime.now().time())
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, (1,state_size,1))
        agent.remember(state, action, reward, next_state, done)
        state = next_state       
        if done:
            agent.update_target_model()
            print('test_done')
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
        watch_result(env.train_index, end_index-start_index, env.get_action(action), reward ,env.profit)
        profit.append(env.profit)
    profit = np.array(profit)
    
    plt.plot(profit, color = 'blue', label = 'Profit')
    plt.title('Profit')
    plt.xlabel('Time')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()
    
        
                      
    