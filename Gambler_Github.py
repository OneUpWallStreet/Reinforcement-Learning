

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import sys
np.set_printoptions(threshold=sys.maxsize)


# In[46]:


MAX_MONEY = 100

discount = 1

final_reward = 1

tran_reward = 0

prob_heads = 0.25


class GamblerProblem:
    
    def __init__(self,size = MAX_MONEY):

        self.state_values = np.zeros((MAX_MONEY+1,))
        self.state_values[MAX_MONEY] = final_reward
        return
        
    def all_states(self):
        return range(MAX_MONEY+1)
    
    def all_actions(self,state=MAX_MONEY+1):
        return range(min(state,MAX_MONEY-state)+1)
    
    def step(self,state,action):
        
        next_state = state+action
        
        if next_state>=MAX_MONEY:
            reward = 1
        
        else:
            reward = 0
            
        return next_state,reward
        
    
    def bellman_expectaion(self,state,action):
        next_state,reward = self.step(state,action)
        return prob_heads*(reward + discount*self.state_values[next_state])
    
    def value_iteration(self,theta = 1e-4):
        
        old_values = []
        count = 0
        
        #For Plotting Delta vs iterations
        delta_plot = []
        
        print('Performing Value Iteration')
        
        while True:
            
            old_values.append(self.state_values.copy())
            
            for state in self.all_states():
                best_value = self.state_values[state]
                for action in self.all_actions(state):
                    
                    new_value = self.bellman_expectaion(state,action)
                    
                    if new_value>best_value:
                        best_value = new_value
                        
                self.state_values[state] = best_value
            
            if count>0:
                delta = np.absolute(np.sum(old_values[count])-np.sum(old_values[count-1]))
                print("Delta {:.5f}".format(delta), end="\r")
                delta_plot.append(delta)

                if delta<theta:
                    print('Found Optimal Value Function in {} steps'.format(count+1))
                    break
                    #We will use break here instead of using "return old_values" to plot Delta versus Iterations
                    #return old_values
                    
            
            count+=1
        
        #Plotting iterations versus Delta
        plt.figure(figsize=(12,8))
        plt.plot(delta_plot, label=".")
        plt.xlabel("Iterations")
        plt.ylabel("Delta")
        plt.title("Delta versus Iterations")
        return old_values
    
    def get_policy(self):
        
        policy = np.zeros((MAX_MONEY+1,))
        
        for state in self.all_states():
            
            best_action = policy[state]
            best_value = self.state_values[state]
            
            for action in self.all_actions(state):
                
                new_value = self.bellman_expectaion(state,action)
                
                if new_value>best_value:
                    best_value = new_value
                    best_action = action
                    
            policy[state] = action
            
        return policy

env = GamblerProblem()
old_values = env.value_iteration()
policy = env.get_policy()
env.state_values
print(policy)
