import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import time
from env import FrozenLake
from env import ACTIONS

BATCH_SIZE = 100
DISCOUNT = 0.75
LR = 0.001

def getkeys(value1,value2):
    # print("This is the value that i got: ",value1.item(),value2.item())
    value = [value1.item(),value2.item()]
    # print("This is the value: ",value)
    for x in ACTIONS:
        if ACTIONS.get(x) == value:
            # print("This is what im giving back")
            return x


class Values:
    def __init__(self):
        self.eps = 0.45
        self.eps_min = 0.05
        self.eps_decay = 0.9997

    def Select_Action(self,train=True):

        sample = np.random.rand()
        self.eps = max(self.eps_min,self.eps*self.eps_decay)

        if train:
            if sample<self.eps:
                action_values = model(env.lake)[0]
                action = torch.argmax(action_values)
                return action.item()

            else:
                action = np.random.randint(0,4)
                return action

        elif train == False:
            action_values = model(env.lake)[0]
            # print("ACtion Values: ",action_values)
            action = torch.argmax(action_values)
            # print("ACtion: ",action)
            return action



class Network(nn.Module):

    def __init__(self):
        super(Network,self).__init__()

        self.hidden1 = nn.Sequential(
            nn.Linear(16,164),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Linear(164,4)
        )


        # self.l1 = nn.Linear(16,164)
        # self.l2 = nn.Linear(164,4)

    def forward(self,x):

        x = x.reshape(1,16)
        x = torch.tensor(x).float()
        # x = F.relu(self.l1(x))
        # x = self.l2(x)
        x = self.hidden1(x)
        x = self.output(x)

        return x


class ReplayMemory(object):

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory)> self.capacity:
            del self.memory[0]

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)

        
def run_episode(env):

    count = 0
    env.reset()
    state = env.lake

    while True:

        count +=1 

        action= V.Select_Action()
        action = ACTIONS.get(action)

        state,action,next_state,reward,done = env.step(env.lake,action)

        trajectory = (state,torch.tensor(action),next_state,torch.tensor(reward),done)

        memory.push(trajectory)

        if done:
            break

        learn()

def train():

    iters = int(input("Number of Iterations: "))
    env.reset()

    for x in range(iters):
        if x % 2 == 0:
            print("Calculating Weights: {:.5f}".format(x), end="\r")

        run_episode(env)

        # if x%100 == 0:
            # print(model.hidden1[0].weight)

def Test_Agent(env):


    env.reset()
    state = env.lake

    while True:
        print("____________________________________________________________")
        action= V.Select_Action(train=False)
        print("ACtion in true: ",action)
        action = ACTIONS.get(action.item())
        print("Action later: ",action)
        print("\n")
        print("Taking Action: ",action)
        print("\n")
        state,action,next_state,reward,done = env.step(env.lake,action)
        print(env.lake)
        print("____________________________________________________________")
        time.sleep(5)

        if done:
            print("Reward is: ",reward)
            break

def learn():

    if len(memory) < BATCH_SIZE:
        return

    tranitions = memory.sample(BATCH_SIZE)

    batch_state,batch_action,batch_next_state,batch_reward,_ = zip(*tranitions)

    # print("BATCH ACTION AT START: ",batch_action[0])

    batch_state = batch_state[0]

    
    batch_action = batch_action[0]
    batch_action = getkeys(batch_action[0],batch_action[1])
    batch_next_state = batch_next_state[0]
    batch_reward = batch_reward[0]

    Q_current_all = model(batch_state)
    Q_current_all = Q_current_all[0]
    
    Q_current = Q_current_all[batch_action]
    # print("BATCH ACTION LATER: ",batch_action)


    Q_next_for_max_action = model(batch_next_state).detach().max(1)[0]
    Q_next_for_max_action = Q_next_for_max_action[0]
    

    Q_expected = batch_reward + (DISCOUNT * Q_next_for_max_action)

    # print("Q_current: ",Q_current)
    # print("Q_expected: ",Q_expected)

    loss = F.smooth_l1_loss(Q_current,Q_expected)


    # print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model = Network()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
env = FrozenLake()
V = Values()


model = Network()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
env = FrozenLake()
V = Values()
print("\n")
print("Weights at start")
print(model.hidden1[0].weight)
train()
print("\n")
print("weights at end")
print(model.hidden1[0].weight)




play = input("Do you want to test?(Y/N): ")

if play == "Y" or play == "y":
    Test_Agent(env)

