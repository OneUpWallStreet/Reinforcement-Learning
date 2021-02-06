import numpy as np
import gym
import sys
import time
np.set_printoptions(threshold=sys.maxsize)

from tilecoding import *

#Importing the OpenAI Gym Enviroment for MountainCar
env = gym.make('MountainCar-v0')

#Removing the episode limit otherwise the enviroment will automatically end if step count goes to 200
env._max_episode_steps = np.inf


"""
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """


ACTIONS = env.action_space.n

print(env.observation_space.low)
print(env.observation_space.high)

POSITION_BOUND = [env.observation_space.low[0],env.observation_space.high[0]]
VELOCITY_BOUND = [env.observation_space.low[1],env.observation_space.high[1]]

POSITION_BOUND

VELOCITY_BOUND

ACTIONS = [0,1,2] 

#STARTING STATE = [POSITION,VELOCITY]


#Tile Coding Value Function made available by Dr Sutton 
#Link: - http://incompleteideas.net/tiles/tiles3.html

class ValueFunction:

    def __init__(self,step_size,number_of_tilings = 8,maxsize = 2048):

        self.number_of_tilings = number_of_tilings
        self.maxsize =maxsize

        self.weights = np.zeros(self.maxsize)

        self.step_size = step_size/self.number_of_tilings

        self.hashable = IHT(self.maxsize)

        self.position_scale = self.number_of_tilings/(POSITION_BOUND[1]-POSITION_BOUND[0])
        self.velocity_scale = self.number_of_tilings/(VELOCITY_BOUND[1]-VELOCITY_BOUND[0])

    def get_active_tiles(self,state,action):

        active_tiles = tiles(self.hashable,self.number_of_tilings,[state[0]*self.position_scale,state[1]*self.velocity_scale],[action])

        return active_tiles

    #Gets the Value of a State-Action pair using the active tiles
    def value(self,state,action):

        if state[0] == POSITION_BOUND[1]:
            return 0 
        else:
            active_tiles = self.get_active_tiles(state,action)

        # print

        return np.sum(self.weights[active_tiles])

    #Updates the weights using TD Learning 
    def Update(self,target,state,action):

        active_tiles = self.get_active_tiles(state,action)

        estimate = np.sum(self.weights[active_tiles])

        delta = self.step_size*(target - estimate)

        for active_tile in active_tiles:
            self.weights[active_tile] += delta

class MountainCar:

    def __init__(self):

        self.eps = 0.1
        # self.eps_decay = 0.997

    def choose_action(self,valueFun):

        num = np.random.rand()

        if num<self.eps:
            return np.random.choice(ACTIONS)

        values = {}

        for a in ACTIONS:
            value = valueFun.value(env.state,a)
            values[a] = value
        # print(values)

        max_action = max(values,key = values.get)
        # print('\nMax_Action: ',max_action)

        return max_action

    def policy(self,valueFun):

        values = {}

        for a in ACTIONS:
            value = valueFun.value(env.state,a)
            values[a] = value

        max_action = max(values,key = values.get)
        return max_action

    def run(self,valueFun,iters = 1):

        for x in range(iters):

            print("Predicting State Values: {:.5f}".format(x), end="\r")
            # self.eps = max(0.1,self.eps*self.eps_decay)

            start_state = env.reset()
            action = self.choose_action(valueFun)

            while True:
                state = env.state
                next_state,reward,done,_ = env.step(action)
                # print('State: ',state)
                # print('Next_State: ',next_state)
                # time.sleep(1)

                if done:
                    valueFun.Update(reward,env.state,action)
                    break

                next_action = self.choose_action(valueFun)
                target = reward + valueFun.value(env.state,next_action)
                # print(target)
                valueFun.Update(target,state,action)

                action = next_action

        # print(valueFun.weights)
        # return valueFun.weights

#Tile Coding with step size as 0.2
stepsize = 0.2
valueFun = ValueFunction(stepsize)

Bike = MountainCar()

#Learning Semi Gradient Sarsa 
iters = 100
Bike.run(valueFun,iters)

#Running the OpenAI enviroment with the optimal policy (without epsilon exploration)
observation = env.reset()
while True:
    env.render()
    print(observation)
    action = Bike.policy(valueFun)
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after timesteps")
        break
env.close()


