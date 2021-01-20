#!/usr/bin/env python
# coding: utf-8

# In[89]:


import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt


# In[2]:


# Race Track from Sutton and Barto Figure 5.6

#Race_Track Environment idea from https://gist.github.com/pat-coady/14978332fce195ea5c1582f49a058f18

big_course = ['WWWWWWWWWWWWWWWWWW',
              'WWWWooooooooooooo+',
              'WWWoooooooooooooo+',
              'WWWoooooooooooooo+',
              'WWooooooooooooooo+',
              'Woooooooooooooooo+',
              'Woooooooooooooooo+',
              'WooooooooooWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWWooooooWWWWWWWW',
              'WWWWooooooWWWWWWWW',
              'WWWW------WWWWWWWW']


tiny_course = ['WWWWWW',
               'Woooo+',
               'Woooo+',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'W--WWW',]


# In[3]:


ACTIONS = [[1,1],[1,0],[1,-1],[-1,-1],[0,-1],[-1,0],[0,0],[0,1],[-1,1]]


# In[76]:


class RaceTrack:
    
    

    def __init__(self):

        self.Velocity_Limit = 5
        self.track = big_course
        self.eps = 1
        self.eps_decay = 0.997
        self.eps_min = 0.05
        self.Velocity_1 = 0
        self.Velocity_2 = 0
        self.nA = len(ACTIONS)
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.Starting_Row = len(self.track)-1
        self.Final_Reward = 0
        self.Move_Reward = -1
        self.N = defaultdict(lambda: np.zeros(self.nA))
        self.real_policy = defaultdict(lambda: 1)

    def reset(self):
        x,y = self.touch_wall_or_reset()
        return x,y

    def get_action_prob(self,Q_S):

        action_prob = np.ones(self.nA)*(self.eps/self.nA)
        best_action = np.argmax(Q_S)
        action_prob[best_action] = (1-self.eps) + (self.eps/self.nA) 
        return action_prob

    def touch_wall_or_reset(self):
  
        Length = len(self.track[self.Starting_Row])

        while True:
            
            num = np.random.randint(Length)
            if self.track[self.Starting_Row][num] == '-':
                x = self.Starting_Row
                y = num
                return x,y

    def step(self,coordinates,action):

        #The X-Coordinate represents the distance from the top while the Y-Coordinate represents the distance from the left
        #Counterintuitive 

        self.Velocity_1 = max(1,min(self.Velocity_1+action[0],self.Velocity_Limit))
        self.Velocity_2 = max(1,min(self.Velocity_2+action[1],self.Velocity_Limit))

        # print('Coordinates: ',coordinates)
        # print('Actions: ', action)
        # print('Velocity_1: ',self.Velocity_1)
        # print('Velocity_2: ',self.Velocity_2)

        x = coordinates[0]
        y = coordinates[1]

        done = False
        if self.track[x][y] == '+':
            done = True
            return (x,y),self.Final_Reward,done

        new_x = x - self.Velocity_1
        #print('New_X = %s,Velocity_1 = %s'%(new_x,self.Velocity_1))
        new_y = y + self.Velocity_2
        #print('New_Y = %s, Velocity_2 = %s '%(new_y,self.Velocity_2))

    

        if new_x >= (len(self.track)) or new_x < 0:
            new_x = x
            #print(new_x)

        if new_y >=(len(self.track[0])) or new_y < 0:
            new_y = y
            #print(new_y)


        next_state =  (new_x,new_y)
        reward = self.Move_Reward

        #print('NextState[0]: %s, NextState[1]: %s'%(next_state[0],next_state[1]))
        if self.track[next_state[0]][next_state[1]] == 'W':
            
          #Moving Back to starting line randomly
          #print('Randomly Moving Back To Finish Line: ',next_state)
            x,y = self.touch_wall_or_reset()
            next_state = (x,y)

        return next_state,reward,done

    def get_episode(self):
        
  
        trajectory = []

        state = self.reset()

        while True:
            

            action_prob =  self.get_action_prob(self.Q[state])
            action_index = np.random.choice(np.arange(self.nA),p=action_prob)
            action = ACTIONS[action_index]
            next_state,reward,done = self.step(state,action)
            trajectory.append((state,action,reward))
            state = next_state
      
            #time.sleep(1)   
            if done:
                break

        return trajectory 

    def Update_Q(self):
        

        tra = self.get_episode()

        states,actions,rewards = zip(*tra)

        for i,state in enumerate(states):

            for j,x in enumerate(ACTIONS):

                if x == actions[i]:
                    action = j

            self.N[state][action] +=1 
            alpha = 1/self.N[state][action]
            self.Q[state][action] += alpha*(sum(rewards[i:]) - self.Q[state][action]) 

        return self.Q

    def best_policy(self):
        
        return dict((state,np.argmax(_action_))for state,_action_ in self.Q.items())

    def best_value(self):
        
        return dict((state,np.max(action))for state,action in self.Q.items())

    def Monte_Carlo_On_Policy(self,iters):

        for i in range(iters):

            self.eps = max(self.eps_min,self.eps*self.eps_decay)

            self.Q = self.Update_Q()

        policy = self.best_policy()
        self.real_policy = self.Real_Actions(policy)

        return self.Q,self.real_policy
    
    def Real_Actions(self,policy):
        states_ava = []
        for i in policy:
            states_ava.append(i)
            
        actions_ava = []
        for state in states_ava:
            action = policy.get(state)
            real_a = ACTIONS[action]
            self.real_policy[state] = real_a 
            
        return self.real_policy


# In[77]:


env = RaceTrack()


# In[85]:


Q,policy = env.Monte_Carlo_On_Policy(iters=10000)


# In[86]:


value = env.best_value()
policy


# In[87]:


Q


# In[88]:


value


# In[75]:


tiny_course[5][1]


# In[ ]:




