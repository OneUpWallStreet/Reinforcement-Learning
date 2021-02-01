import numpy as np
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.colors import LinearSegmentedColormap

W = LinearSegmentedColormap.from_list('w', ["w", "w"], N=256)

ACTIONS = {
    0: [1, 0],   # south
    1: [-1, 0],  # north
    2: [0, -1],  # west
    3: [0, 1],   # east
}

class Maze:

    def __init__(self):
        np.random.seed(0)
        self.x = 9
        self.real_x = 8
        self.y = 6
        self.real_y = 5
        self.nA = len(ACTIONS)
        self.Final_Reward = 1
        self.Move_Reward = 0
        self.eps = 1
        self.eps_decay = 0.9997
        self.n = 30
        self.alpha = 0.002
        self.discount = 0.97
        self.eps_min = 0.05
        self.final_state = (1,8)
        self.state_values = np.zeros((self.y,self.x))
        self.policy_values = np.zeros((self.y,self.x),dtype=str)
        self.policy_values = np.zeros((self.y,self.x),dtype=str)

        self.Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
        self.feature_size = 4
        self.features = defaultdict(lambda: np.zeros(self.feature_size))
        self.w = defaultdict(lambda: np.zeros(self.feature_size))



    def reset(self):
        state = (0,2)
        return state

    def step(self,state,action):
        done = False
        if state == (0,8):
            done = True
            reward = self.Final_Reward
            return state,reward,done

        next_state = (state[0]+action[0],state[1]+action[1])
        reward = self.Move_Reward

        if next_state == (1,2) or next_state == (2,2) or next_state == (3,2) or next_state == (4,5) or next_state == (0,7) or next_state == (1,7) or next_state == (2,7):
            next_state = state

        if next_state[0]<0 or next_state[0]>=self.y:
            next_state = state

        if next_state[1]<0 or next_state[1]>=self.x:
            next_state = state 

        return next_state,reward,done


    def get_action_prob(self,Q_S):

        action_probs = np.ones(self.nA)*(self.eps/self.nA)
        best_action = np.argmax(Q_S)
        action_probs[best_action] += (1-self.eps)

        return action_probs
    
    def get_action(self,state):

        action_prob = self.get_action_prob(self.Q[state])
        _action_ = np.random.choice(np.arange(self.nA),p = action_prob) if state not in self.Q else np.random.randint(0,4)
        # action = ACTIONS.get(_action_)

        return _action_

    def best_policy(self):

        return dict((state, np.argmax(actions)) for state,actions in self.Q.items())

    def best_value(self):

        return dict((state, np.max(actions)) for state,actions in self.Q.items())


    def Semi_Gradient_Sarsa_Control(self):

        state = self.reset()

        _action_ = self.get_action(state)

        action = ACTIONS.get(_action_)

        while True:

            next_state,reward,done = self.step(state,action)

            state_action = (state,_action_)

            self.features[state_action] = [self.real_y-state[0],self.real_x-state[1],state[0],state[1]]
            
            f_arr_s_a = np.array(self.features[state_action])
            
            w_arr_s_a = np.array(self.w[state_action])

            # print(f_arr_s_a)
            # print(w_arr_s_a) 

            

            State_Value_Current = np.dot(w_arr_s_a,f_arr_s_a)

            # print(State_Value_Current)
            # time.sleep(0.2)

            self.Q[state_action[0]][state_action[1]] = State_Value_Current

            _action_= self.get_action(next_state)
            action = ACTIONS.get(_action_)

            next_state_next_action = (next_state,_action_)
            f_arr_n_s_a = np.array(self.features[next_state_next_action])
            w_arr_n_s_a = np.array(self.w[next_state_next_action])

            Next_State_Value = np.dot(w_arr_n_s_a,f_arr_n_s_a)
            

            if done:
                for i,x in enumerate(self.features[state_action]):
                    self.w[state_action][i] += self.alpha*(reward - State_Value_Current)*self.features[state_action][i]

            for i,x in enumerate(self.features[state_action]):
                self.w[state_action] += self.alpha*(reward + self.discount*(Next_State_Value) - State_Value_Current)*self.features[state_action][i]


            if done:
                break
            
            state = next_state

        return self.w,self.features


    def loop(self):

        for x in range(1000):
            print("Predicting State Values: {:.5f}".format(x), end="\r")
            self.eps = max(self.eps_min,self.eps*self.eps_decay)
            self.w,self.features = self.Semi_Gradient_Sarsa_Control()
            
        policy_initial = self.best_policy()
        policy = self.Real_Actions(policy_initial)

        value_initial = self.best_value()
        self.state_values = self.Real_Value(value_initial)
        self.render()

        return self.w,self.features,self.Q,policy,self.state_values


    def render(self, title=None):
        """
        Displays the current value table of mini gridworld environment
        """
        size = len(self.state_values) if len(self.state_values) < 20 else 20
        fig, ax = plt.subplots(figsize=(self.x, self.y))
        if title is not None:
            ax.set_title(title)
        ax.grid(which='major', axis='both',
                linestyle='-', color='k', linewidth=2)
        sn.heatmap(self.state_values, annot=True, fmt=".1f", cmap=W,
                   linewidths=1, linecolor="black", cbar=False)
        plt.show()
        return fig, ax


    def Real_Actions(self,policy):
        y = []

        for x in policy:
            y.append(x)

        policy_actions = {}

        for x in y:

            action = policy.get(x)
   
            if action == 0:
                policy_actions.update({x:'South'})
            if action  == 1:
                policy_actions.update({x:'North'})
            if action  == 2:
                policy_actions.update({x:'West'})
            if action  == 3:
                policy_actions.update({x:'East'})

        return policy_actions


    def Real_Value(self,value):

        for x in value.items():
            self.state_values[x[0]] = x[1]

        return self.state_values

env = Maze()


w,f,Q,policy,state_values = env.loop()

print(state_values)


print(Q)

print(policy)




