import gymnasium as gym
from Env import Env
import cv2
import torch

env = gym.make("DemonAttackDeterministic-v4")
env = Env(env)

state = env.reset()
r = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)
cv2.imshow("r.png",r)
cv2.waitKey(0)

def choose_action(state):
    print(state.shape)
    state = cv2.cvtColor(state,cv2.COLOR_BGR2GRAY)
    state = torch.tensor(state,dtype = torch.float32)
    state = state.reshape(1,420,322)

print(choose_action(state)) 