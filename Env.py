import numpy as np
import cv2
from collections import deque

line = np.ones((210,2,3),dtype = np.uint8)
line = line*255
cv2.imwrite("data\images\line.jpg",line)
line = cv2.imread("data\images\line.jpg")

class Env:
    def __init__(self,env):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.buffer = deque(maxlen=4)
        self.randomness()

    def randomness(self):
        for i in range(4):
            self.buffer.append(self.env.reset()[0])

    def reset(self):
        reset = self.env.reset()
        self.buffer.append(reset[0])
        image1 = cv2.hconcat([self.buffer[-4],line,self.buffer[-3]])
        image2 = cv2.hconcat([self.buffer[-2],line,self.buffer[-1]])
        image  = cv2.vconcat([image1,image2])
        save   = cv2.imwrite("data\images\concat-image.jpg",image)
        return image
    
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def next_state(self,new_state):
        self.buffer.append(new_state)
        image1 = cv2.hconcat([self.buffer[-4],line,self.buffer[-3]])
        image2 = cv2.hconcat([self.buffer[-2],line,self.buffer[-1]])
        image  = cv2.vconcat([image1,image2])
        save   = cv2.imwrite("data\images\concat-next-image.jpg",image)
        return image
    
    def step(self,action):
        next_state,reward,done,info,_ = self.env.step(action)
        next_state = self.next_state(next_state)
        return next_state,reward,done,info