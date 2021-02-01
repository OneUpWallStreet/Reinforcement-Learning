import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

W = LinearSegmentedColormap.from_list('w', ["w", "w"], N=256)

ACTIONS = {
    0: [1, 0],   
    1: [-1, 0],  
    2: [0, -1],  
    3: [0, 1],}

#ACTUAL STATE VALUES FROM DYNAMIC PROGRAMMING
# [[  0. -14. -20. -22.]
#  [-14. -18. -20. -20.]
#  [-20. -20. -18. -14.]
#  [-22. -20. -14.   0.]]

class GridWorld:
    def __init__(self, size=4):

        self.size = size
        self.state_value = np.zeros((self.size,self.size))
        self.feature_size = 4
        self.state_space_size = self.size*self.size
        self.real_size = self.size -1
        
        # self.w = np.ones((self.state_space_size,self.feature_size))
        # self.feature = np.ones((self.state_space_size,self.feature_size))

        #Features = [Direction From 3 - X Axis, Direction From 3 - Y Axis,Direction From 0 - X Axis, Direction From 0 - Y Axis,Current X, Current Y]
        self.features = defaultdict(lambda: np.zeros(self.feature_size))
        self.w = defaultdict(lambda: np.zeros(self.feature_size))
        self.alpha = 0.001
        self.discount = 1
        return


    def reset(self):
        # self.state_value = np.zeros((self.size, self.size))
        x = np.random.randint(self.size)
        y = np.random.randint(self.size)

        state = (x,y)
    
        return state

    def step(self, state, action):
        done = False
        # is terminal state?
        size = len(self.state_value) - 1
        if (state == (0, 0)) or (state == (size, size)):
            done = True
            return state, 0,done

        s_1 = (state[0] + action[0], state[1] + action[1])
        reward = -1
        # out of bounds north-south
        if s_1[0] < 0 or s_1[0] >= len(self.state_value):
            s_1 = state
        # out of bounds east-west
        elif s_1[1] < 0 or s_1[1] >= len(self.state_value):
            s_1 = state

        return s_1, reward,done

    def get_action(self):

        _action_ = np.random.randint(4)
        action = ACTIONS.get(_action_)

        return action

    def TD_Semi_Gradient(self):
    #Features = [Direction From 3 - X Axis, Direction From 3 - Y Axis,Direction From 0 - X Axis, Direction From 0 - Y Axis,Current X, Current Y]
        
        state = self.reset()

        while True:

            action = self.get_action()

            next_state,reward,done = self.step(state,action)

            self.features[state] = [self.real_size-state[0],self.real_size-state[1],state[0],state[1]]

            f_arr_s = np.array(self.features[state])
            w_arr_s = np.array(self.w[state])

            f_arr_n_s = np.array(self.features[next_state])
            w_arr_n_s = np.array(self.w[next_state])


            Value_State = np.dot(w_arr_s,f_arr_s)
            Value_Next_State = np.dot(w_arr_n_s,f_arr_n_s)

            if done:
                Value_Next_State = 0

            for i,real_feature in enumerate(self.features[state]):

                self.w[state][i] += self.alpha*(reward + self.discount*(Value_Next_State) - Value_State)*self.features[state][i]


            if done:
                break

            state = next_state


        return self.w,self.features

    def loop(self):


        for x in range(200000):

            print("Predicting State Values:  {:.5f}".format(x), end="\r")
            self.w,self.features = self.TD_Semi_Gradient()


        return self.w,self.features

env = GridWorld()


w,f = env.loop()

state_value = np.zeros((env.size,env.size))
print('\n Initial State Values...')
print(state_value)

for state in f:
    # print(f[state])
    f_arr = np.array(f[state])
    w_arr = np.array(w[state])
    # print(w_arr)

    val = np.dot(w_arr,f_arr)
    state_value[state] = val
    # print(val)

print('\n')
print('Final State Values...')
print(state_value)

