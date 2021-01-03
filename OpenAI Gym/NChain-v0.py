import gym
import numpy as np

#Creating States and Actions
states = []
for x in range(5):
  states.append(x)
  
ACTIONS = (0,1)

def value_iteration(env,theta = 0.9,lmda = 0.9):

  state_values = np.zeros((len(states)))
  new_state_values = np.zeros((len(states)))
  i = 0

  while True:

    i+=1

    for _state_ in range(len(states)):

      a_state_value = 0
      action_values = []

      for action in ACTIONS:

        next_state,reward,done,_ = env.step(action)

        a_state_value = 0.5*(reward+lmda*state_values[next_state])

        action_values.append(a_state_value) 

      best_action = np.argmax(action_values)

      new_state_values[env.state] = action_values[best_action]


    delta = np.absolute(np.sum(new_state_values-state_values))
    state_values = new_state_values.copy()
    
    if i > 10000:

      if delta<theta:
        print('Found the best value function')
        print('Delta: ',delta)
        break

  return state_values
  
#Creating Gym enviroment
env = gym.make('NChain-v0')
env.reset()

state_values = value_iteration(env,theta = 1e-4,lmda = 0.9)
print(state_values)

def get_policy(env,state_values,lmda=0.9):

  policy = np.zeros((len(states)))
  print('arbitrary policy')
  print(policy)

  for _state_ in range(len(states)):

    env.state = _state_
    action_values = []

    for action in ACTIONS:

      next_state,reward,done,_ = env.step(action)

      a_state_value = 0.5*(reward+ lmda*state_values[next_state])
      action_values.append(a_state_value)

    best_action = np.argmax(action_values)

    policy[env.state] = best_action

  return policy

policy = get_policy(env,state_values)

print('Final Policy')
print(policy)

def test_run(env,policy):

  done = False
  env.state = env.reset()
  total_reward = 0
  total_steps = 0
  while done!=True:
    
    #Converting to int to avoid error :D
    action = int(policy[env.state])
    
    env.state_,reward,done,_ = env.step(action)
    
    

    total_reward += reward
    total_steps +=1 

  return total_reward,total_steps
  
reward,step = test_run(env,policy)

print(step)
print(reward)
