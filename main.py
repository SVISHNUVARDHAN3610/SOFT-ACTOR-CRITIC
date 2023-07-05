import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import gymnasium as gym

from torch.autograd import Variable 
from Env import Env
from PolicyNet import PolicyNet
from ValueNet import ValueNet
from QNet import QNet
from ReplayBuffer import Replay_Buffer
 
main_env = gym.make("DemonAttackDeterministic-v4")
env = Env(main_env)

class Main:
    def __init__(self,state_size,action_size,faltten_size,n_games,steps,batch_size,maxlen,gamma,lamda,load,paths):
        self.state_size  = state_size
        self.action_size = action_size
        self.flatten_size= faltten_size
        self.n_games     = n_games
        self.steps       = steps
        self.batch_size  = batch_size
        self.maxlen      = maxlen
        self.gamma       = gamma
        self.lamda       = lamda
        self.load        = load
        self.paths       = paths
        self.buffer      = Replay_Buffer(self.batch_size,self.maxlen)
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net  = PolicyNet(self.state_size,self.action_size,self.flatten_size,self.device)
        self.value_net   = ValueNet(self.state_size,self.flatten_size,self.device)
        self.q_net       = QNet(self.state_size,7,self.flatten_size,self.device)
        self.policy_optim= optim.Adam(self.policy_net.parameters(),lr = 0.00075)
        self.value_optim = optim.Adam(self.value_net.parameters(),lr = 0.00085)
        self.q_optim     = optim.Adam(self.q_net.parameters() ,lr =0.0005)
        self.rewards     = []
        self.policy_loss = []
        self.value_loss  = []
        self.csv         = csv.writer(open("data\csv\main.csv",'w'))

    def choose_action(self,state):
        state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)
        state = torch.tensor(state,dtype = torch.float32).to(self.device)
        state = state.reshape(1,420,322)
        dict  = self.policy_net(state)
        return dict
    
    def q_value(self,state,action):
        state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)
        state = torch.tensor(state,dtype = torch.float32).to(self.device)
        state = state.reshape(1,420,322)
        action= torch.tensor(action).float().to(self.device)
        q_value= self.q_net(state,action)
        return q_value
    
    def value(self,state):
        state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)
        state = torch.tensor(state,dtype = torch.float32).to(self.device)
        state = state.reshape(1,420,322)
        value = self.value_net(state)
        return value
    
    def optim_update(self,policy_loss,value_loss):
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        self.q_optim.zero_grad()
        self.value_optim.zero_grad()
        value_loss.backward(retain_graph=True)
        self.q_optim.step()
        self.value_optim.step()

    def load_and_save(self):
        if self.load:
            self.policy_net.load_state_dict(torch.load(self.paths[0]))
            self.q_net.load_state_dict(torch.load(self.paths[1]))
            self.value_net.load_state_dict(torch.load(self.paths[2]))
        torch.save(self.policy_net.state_dict(),self.paths[0])
        torch.save(self.q_net.state_dict(),self.paths[1])
        torch.save(self.value_net.state_dict(),self.paths[2])

    def appending(self,policy_loss,value_loss):
        self.policy_loss.append(policy_loss.detach().numpy())
        self.value_loss.append(value_loss.detach().numpy())

    def update(self,m_state,m_action):
        if len(self.buffer.states) > self.batch_size:
            states,next_states,rewards,terminals,actions = self.buffer.sample()
            for i in range(self.batch_size):
                next_state = next_states[i]   
                next_value = self.value(next_state)             
                q_value = self.q_value(states[i],actions[i])
                value_target  = 0.5*q_value -(rewards[i]+self.lamda*next_value)**2
                value_loss = value_target-q_value
                value_loss = Variable(value_loss,requires_grad=True)
                #policy_loss
                policy_loss = torch.log(torch.tensor(actions[i]))*self.gamma - q_value
                policy_loss = policy_loss.mean()
                policy_loss = Variable(policy_loss,requires_grad= True)
                print("policy_loss",value_target)
                self.appending(policy_loss,value_loss)
                self.optim_update(policy_loss,value_loss)
                self.load_and_save()
            
            return value_loss,policy_loss
        else:
            return 0,0,
            
    def action_updation(self,action):
        updated = []
        for i in range(6):
            updated.append(action[i].item())
        updated.append(action.argmax(0).item())  
        return updated
    
    def ploting(self):
        if len(self.buffer.states)>self.batch_size:
            plt.plot(self.rewards,color='red',linestyle='dashed')
            plt.xlabel("epochs")
            plt.ylabel("rewards")
            plt.title("Rewards Graph")
            plt.savefig("data/images/rewards.png")
            plt.close()
            ###
            plt.plot(self.policy_loss,color="blue",label="policy-loss")
            plt.plot(self.value_loss,color="orange",label="value-loss")
            plt.xlabel("epochs")
            plt.ylabel("losses")
            plt.legend()
            plt.savefig("data/images/losses.png")
            plt.close()
    def train(self):
        for i in range(self.n_games):
            rewards,policy_losss,value_losss,count = 0,0,0,0
            state = env.reset()            
            for j in range(self.steps):
                dict = self.choose_action(state)
                action = dict.argmax(0).item()
                next_state,reward,done,info = env.step(1)
                next_state = next_state
                action = self.action_updation(dict)
                self.buffer.append(state,next_state,reward,done,action)
                rewards += reward      
                
                if not done:
                    value_loss,policy_loss = self.update(state,action)
                    policy_losss+=policy_loss
                    value_losss+=value_loss
                    state = next_state 
                    self.ploting()            
                else:
                    print("pass")

            self.rewards.append(rewards)
            print(f'episode:{i} rewards:{rewards} policy_loss:{policy_losss} value_losses {value_losss}')

