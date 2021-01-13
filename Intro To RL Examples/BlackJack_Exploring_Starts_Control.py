#!/usr/bin/env python
# coding: utf-8

# In[46]:


import gym
import numpy as np
from collections import defaultdict
import plot_utils
import gym
from IPython import display
import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_blackjack_values(V):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()

def plot_policy(policy):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in policy:
            return policy[x,y,usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x,y,usable_ace) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0,1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)','1 (HIT)'])
            
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


# In[48]:


env = gym.make('Blackjack-v0')
discount =1


# In[49]:


def random_policy(state):
    
    prob = np.random.rand()
    
    if prob<0.5:
        return 0
    
    else:
        return 1


# In[50]:


def get_trajectory(policy,env):
    
    state = env.reset()
    trajectory = []
    
    
    while True:
        action = random_policy(state)
        next_state,reward,done,_ = env.step(action)
        trajectory.append((state,action,reward))
        state = next_state
        
        if done:
            break
            
    
    return trajectory    


# In[51]:


def best_policy(Q):
    
    return dict((k,np.argmax(v)) for k,v in Q.items())


# In[52]:


def update_Q(env,tra,Q,G,old_state_actions,N):
    
    states,actions,rewards = zip(*tra)

        
        
    for t in range(len(states)-1,-1,-1):
            
        R = rewards[t]
        S = states[t] 
        A = actions[t]
            
        S_A = (S,A)
            
        G += R
            
        if S_A not in old_state_actions:
                
            N[S_A] +=1
            Q[S][A] += (G - Q[S][A])/N[S_A]
            old_state_actions.append(S_A)
        


# In[53]:


def MC_Control(env,iters=500000):
    
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    N = defaultdict(float)
    
    for _ in range(iters):
        
        old_state_actions = []
        G = 0
        
        tra = get_trajectory(random_policy,env)
        
        update_Q(env,tra,Q,G,old_state_actions,N)
        
        policy = best_policy(Q)
        
        
    return Q,policy


# In[63]:


Q,policy = MC_Control(env,iters=500000)
V = dict((k,np.max(v)) for k, v in Q.items())


# In[64]:


plot_blackjack_values(V)


# In[65]:


plot_policy(policy)


# In[68]:


V


# In[ ]:





# In[ ]:




