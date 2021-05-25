import numpy as np
from collections import defaultdict

ACTIONS = {
    0: [0,1],
    1: [1,0],
    2: [-1,0],
    3: [0,-1]
}

class FrozenLake:

    def __init__(self):
        np.random.seed(10)
        self.reset()
        self.discount = 0.7
        self.alpha = 0.02
        self.final_reward = 10
        self.step_reward = -0.1
        self.fall_reward = -10
        self.prob_slip = 0.25  

        self.eps = 0.4
        self.eps_min = 0.05
        self.eps_decay = 0.99997


        self.count = 0
        self.nA = len(ACTIONS)



    def reset(self):
        self.lake = np.ones((4,4))
        self.lake[0][0] = 7
        self.lake[3][3] = 9
        self.lake[0][2] = 0
        self.lake[3][0] = 0
        self.lake[1][3] = 0
        # print(self.lake)


    def step(self,current_state,action):

        position = np.where(current_state == 7)

        state = (position[0].item(),position[1].item())

        self.count +=1

        done = False

        prob = np.random.rand()

        prob = 1

        
        #Small Chance you slip 
        if prob<self.prob_slip:
            _action_ = np.random.randint(0,4)
            action = ACTIONS.get(_action_)

        next_state = (state[0] + action[0],state[1] + action[1])

    
        #North South Conditions
        if next_state[0] > 3 or next_state[0] < 0:
            next_state = (state[0],state[1])

        #East-West Conditions
        if next_state[1] > 3 or next_state[1] < 0:
            next_state = (state[0],state[1])


        if self.lake[next_state] == 0:

            done = True
            reward = self.fall_reward
            next_state = self.lake
            return current_state,action,next_state,reward,done

        #Final Step i.e. game ends
        if(next_state == (3,3)):
            done = True  
            reward = self.final_reward
            next_state = self.lake
            return current_state,action,next_state,reward,done

# and self.lake[next_state]!=0
        if next_state != state and self.lake[next_state]!=0:
            
            self.lake[state[0]][state[1]] = 1

            self.lake[next_state[0]][next_state[1]] = 7

        done = False
        reward = self.step_reward

        next_state = self.lake

        return current_state,action,next_state,reward,done