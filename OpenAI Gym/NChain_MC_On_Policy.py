import gym
import numpy as np
from collections import defaultdict

def best_policy(Q):
    
    return dict((state,np.argmax(best_action)) for state,best_action in Q.items())

def best_value(Q):
    
    return dict((state,np.max(action)) for state,action in Q.items())

env = gym.make('NChain-v0')


def get_action_prob(Q_s,state,nA,eps):
    
    action_prob = np.ones(nA)*(eps/nA)
    best_action = np.argmax(Q_s)
    action_prob[best_action] = (1-eps) + (eps/nA)
    return action_prob


def get_trajectory(Q,eps,nA):
    
    state = env.reset()
    trajectory = []
    
    while True:
        
        action_prob = get_action_prob(Q[state],state,nA,eps) 
        action = np.random.choice(np.arange(nA),p=action_prob) if state in Q else env.action_space.sample()
        
        next_state,reward,done,_ = env.step(action)
        trajectory.append((state,action,reward))
        state = next_state
        if done:
            break
        
    return trajectory

def update_Q(tra,Q,alpha):
    
    states,actions,rewards = zip(*tra)
    
    for i,state in enumerate(states):
        
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha*(sum(rewards[i:]) - old_Q)
        
    return Q

def MC_Control(alpha,eps_start,eps_min,eps_decay,Q,nA,iters):
    
    eps = eps_start
    for _ in range(iters):
        
        eps = max(eps_min,eps*eps_decay)
        tra = get_trajectory(Q,eps,nA)
        Q = update_Q(tra,Q,alpha)
        policy = best_policy(Q)
    return Q,policy

nA = env.action_space.n
Q = defaultdict(lambda: np.zeros(nA))
eps_start = 1
eps_min = 0.05
eps_decay = 0.995
iters = 10
alpha = 0.02

Q,policy = MC_Control(alpha,eps_start,eps_min,eps_decay,Q,nA,iters)
V = best_value(Q)

print(Q)
print(V)
print(policy)



