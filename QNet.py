import torch
import torch.nn as nn
import torch.nn.functional as f

class QNet(nn.Module):
  def __init__(self,state_size,action_size,flatten_size,device):
    super(QNet,self).__init__()
    self.satate_size = state_size
    self.action_size = action_size
    self.flatten_size = flatten_size
    self.conv1       = nn.Conv2d(in_channels=1,out_channels=9,kernel_size = 3 ,stride = 1).to(device)
    self.conv2       = nn.Conv2d(in_channels=9,out_channels=15,kernel_size = 3 ,stride = 1).to(device)
    self.conv3       = nn.Conv2d(in_channels=15,out_channels=32,kernel_size = 3 ,stride = 1).to(device)
    self.act         = nn.Linear(self.action_size,32).to(device)
    self.lin         = nn.Linear(self.flatten_size,32).to(device)
    self.max         = nn.MaxPool2d(3)
    self.lin1        = nn.Linear(64,32)
    self.lin2        = nn.Linear(32,1).to(device)
    self.init_weights()
  def forward(self,x,action):
    x = self.conv1(x)
    x = self.max(x)
    x = self.conv2(x)
    x = self.max(x)
    x = self.conv3(x)
    x = self.max(x)
    state = x.reshape(-1,)
    action = self.act(action)
    state  = self.lin(state)
    x      = torch.cat([state,action])
    x      = f.relu(self.lin1(x))
    x      = f.relu(self.lin2(x))
    return x
  def init_weights(self):
    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
      elif isinstance(m,nn.Linear):
        nn.init.xavier_uniform(m.weight)