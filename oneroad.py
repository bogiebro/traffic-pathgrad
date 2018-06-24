import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import rg
from rg import mc
import numpy as np
from gumbel import gumbel_softmax

# IDM params
T = 2.0
B = 6.0
L = 4.5
V0 = 20.0
S0 = 1.5
DELTA = 4.0
A = 3.5

# Globals
PARAMS = 2
xi,vi = range(PARAMS)
YELLOW_TICKS = 6
EPS = 1e-9
RATE = 0.5
SCALE = 8
DSIZE = 20
SQRT_AB = 2 * np.sqrt(A*B)
RL = 250
INF = 1000000
THRESH = 220

# Main IDM computation. Car me follows car ld
def sim(ld, me):
  v = me[vi]
  s_star = S0 + F.relu(v*T + v * (v - ld[vi]) / SQRT_AB)
  s = ld[xi] - me[xi] - L
  dv = A * (1 - (v / V0)**DELTA - (s_star / (s + EPS))**2)
  dvr = dv*RATE
  dx = RATE*v + 0.5*dvr*RATE
  return torch.stack((me[xi] + F.relu(dx), F.relu(v + dvr)))

# Light is one hot in R2: red prob, green prob
light_vec = Variable(torch.Tensor([RL, INF]))
def set_light(road, light):
  road[0] = light_vec.dot(road)

def advance_finished_cars(r):
  advanced_xs = r[xi] > RL
  advanced_xs[0] = 0
  if advanced_xs.data.any():
    advanced = advanced_xs.expand_as(r)
    return torch.masked_select(r, (1 - advanced.float()).byte()).view(PARAMS, -1)
  return r

# Can we differentiate through this?
def queue_length(r):
  return (r[1:] > THRESH).sum()

class IDM:
  def __init__(self):
    self.road = Variable(torch.Tensor([[RL], [0]]))
    self.steps = 0
    self.next_add = 0
    self.speedv = Variable(torch.Tensor([11.11]))

  def step(self):
    while self.steps >= self.next_add:
      self.next_add += np.random.exponential(scale=SCALE)
      x0,_ = torch.min(self.road[xi,-1] - L - S0, 0)
      carv = torch.cat([x0, self.speedv])
      self.road = torch.cat((self.road, carv.unsqueeze(1)), 1)
    if self.road.size(1) > 1: 
      road = torch.cat((self.road[0:1], sim(self.road[:-1], self.road[1:])), 1)
    self.road = advance_finished_cars(self.road)
    self.steps += 1

# Simple 2 layer nn
class IDMTrainer(nn.Module):
  def __init__(self, model, opts):
    super().__init__()
    self.model = model
    self.fc1 = nn.Linear(1, 4)
    self.fc2 = nn.Linear(4, 2)
  def forward(self, x, temp=1):
    logits = self.fc2(F.relu(self.fc1(x)))
    result = gumbel_softmax(logits, temp, hard=True)
    return logits, result

def main():
  reward = Variable(torch.zeros(1))
  for i in count(0):
    # Do this.
# Observations: length of queue?
# Reward: negative queue length.
# Let's do this!

# First obvious thing: have a single road and a single stoplight. 
# We should obviously learn to turn the stoplight green

# Let's just have fun here. 
# Start with just the stoplights in your list.
# As you add cars, also add them to a mask
# Stoplights turn into cars when the car passes them
