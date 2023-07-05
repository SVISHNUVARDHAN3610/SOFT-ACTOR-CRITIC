from collections import deque
import random

class Replay_Buffer:
  def __init__(self,batch_size,maxlen):
    self.batch_size  = batch_size
    self.maxlen      = maxlen
    self.states      = deque(maxlen = self.maxlen)
    self.next_states = deque(maxlen = self.maxlen)
    self.rewards     = deque(maxlen = self.maxlen)
    self.terminals   = deque(maxlen = self.maxlen)
    self.actions     = deque(maxlen = self.maxlen)
  def append(self,state,next_state,reward,done,action):
    self.states.append(state)
    self.next_states.append(next_state)
    self.rewards.append(reward)
    self.terminals.append(done)
    self.actions.append(action)
  def sample(self):
    rand = random.randint(0,len(self.states)-1)
    states,next_states,rewards,terminals,actions = [],[],[],[],[]
    for i in range(self.batch_size):
      state      = self.states[rand]
      next_state = self.next_states[rand]
      reward     = self.rewards[rand]
      done       = self.terminals[rand]
      action     = self.actions[rand]
      states.append(state)
      next_states.append(next_state)
      rewards.append(reward)
      terminals.append(done)
      actions.append(action)
    return states,next_states,rewards,terminals,actions