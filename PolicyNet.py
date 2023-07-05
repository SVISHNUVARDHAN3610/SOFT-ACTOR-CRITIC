import torch
import torch.nn as nn
import torch.nn.functional as f

class PolicyNet(nn.Module):
  def __init__(self,state_size,action_size,flatten_size,device):
    super(PolicyNet,self).__init__()
    self.state_size  = state_size
    self.action_size = action_size
    self.flatten_size= flatten_size
    self.conv1       = nn.Conv2d(in_channels=1,out_channels=9,kernel_size = 3 ,stride = 1).to(device)
    self.conv2       = nn.Conv2d(in_channels=9,out_channels=15,kernel_size = 3 ,stride = 1).to(device)
    self.conv3       = nn.Conv2d(in_channels=15,out_channels=32,kernel_size = 3 ,stride = 1).to(device)
    self.max         = nn.MaxPool2d(3)
    self.lin1        = nn.Linear(self.flatten_size,128).to(device)
    self.lin2        = nn.Linear(128,64).to(device)
    self.lin3        = nn.Linear(64,32).to(device)
    self.lin4        = nn.Linear(32,self.action_size).to(device)
    self.tanh        = nn.Tanh()
    self.init_weights()
  def forward(self,x):
    x = self.conv1(x)
    x = self.max(x)
    x = self.conv2(x)
    x = self.max(x)
    x = self.conv3(x)
    x = self.max(x)
    x = x.reshape(-1,)
    x = f.relu(self.lin1(x))
    x = f.relu(self.lin2(x))
    x = f.relu(self.lin3(x))
    x = f.softmax(self.lin4(x))
    return x
  def init_weights(self):
    for m in self.modules():
      if isinstance(m,nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
      elif isinstance(m,nn.Linear):
        nn.init.xavier_uniform(m.weight)