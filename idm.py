import torch
from torch.autograd import Variable
import torch.nn.functional as F
import rg
from rg import mc
import numpy as np

# What now?
# We could base it off of something else

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
THRESH = 230
DSIZE = 20
SQRT_AB = 2 * np.sqrt(A*B)
VTHRESH = 0.2
OBS_LEN = 40

# Update more frequently when we're animating so it's less jerky
def set_render_rate():
  global RATE
  global SCALE
  RATE *= 0.1
  SCALE *= 10

# Main IDM computation. Car me follows car ld
def sim(ld, me):
  v = me[vi]
  s_star = S0 + F.relu(v*T + v * (v - ld[vi]) / SQRT_AB)
  s = ld[xi] - me[xi] - L
  dv = A * (1 - (v / V0)**DELTA - (s_star / (s + EPS))**2)
  dvr = dv*RATE
  dx = RATE*v + 0.5*dvr*RATE
  return torch.stack((me[xi] + F.relu(dx), F.relu(v + dvr)))

# We maintain for each road segment a torch tensor of cars.
# The first element is a fake car: the stoplight's loc (if it's red) or the car in the
# next road (if it's green) or 1e3 (if there is no next car). 
# New cars are always added at the back of the list
class IDM:
  def __init__(self, graph):
    self.graph = graph
    self.lensv = Variable(mc(torch.Tensor([self.graph.len])).expand(self.graph.roads))
    self.probs = mc(torch.Tensor([0.5])).expand(self.graph.intersections)
    self.zerov = Variable(mc(torch.zeros(1)))
    self.flipv = Variable(torch.Tensor([[0,1],[1,0]]))
    self.speedv = Variable(torch.Tensor([11.11]))

  def random_phase(self):
    lights = mc(torch.bernoulli(self.probs))
    return torch.stack((lights, 1 - lights),1)

  def set_phase(self, current_phase):
    self.update_lights(current_phase)
    self.elapsed = self.elapsed * (current_phase[:,0] * self.previous_phase[:,0])
    self.elapsed += 1
    self.choices += 1
    self.previous_phase = current_phase

  def step(self):
    while self.steps >= self.next_add:
      self.next_add += np.random.exponential(scale=SCALE)
      rd = np.random.choice(self.graph.entrypoints)
      x0,_ = torch.min(self.cars[rd][xi,-1] - L - S0, 0)
      carv = torch.cat([x0, self.speedv])
      self.add_car(rd, carv)
    for (i,c) in enumerate(self.cars):
      if c.size(1) > 1: 
        self.cars[i] = torch.cat((c[:,0:1], sim(c[:,:-1],c[:,1:])), 1)
    self.advance_finished_cars()
    self.steps += 1
    self.update_leading_cars()

  def update_leading_cars(self):
    nextcars = torch.cat([self.graph.len + self.cars[self.graph.nexts[i]][xi,-1:]
     for i in range(self.graph.roads)])
    self.stoplightv = torch.stack([self.lensv, nextcars], 1)

  def reset(self):
    self.score = Variable(torch.zeros(1))
    self.steps = 0 # Number of simulator ticks
    self.choices = 0 # Number of times we chose the light color
    self.next_add = 0 # Tick at which to next add a car
    self.cars = [Variable(mc(torch.Tensor(PARAMS, 1))) for _ in range(self.graph.roads)]
    for c in self.cars:
      c[vi,0] = 0
      c[xi,0] = 1e3
    self.elapsed = Variable(mc(torch.zeros(self.graph.intersections)))
    self.previous_phase = Variable(self.random_phase())
    self.rewards = Variable(mc(torch.zeros(1)))
    self.update_leading_cars()

  def add_car(self, road, car):
    self.cars[road] = torch.cat((self.cars[road], car.unsqueeze(1)), 1)

  def advance_finished_cars(self):
    for (i,r) in enumerate(self.cars):
      advanced_xs = r[xi] > self.graph.len
      advanced_xs[0] = 0
      if advanced_xs.data.any():
        advanced = advanced_xs.expand_as(r)
        self.cars[i] = torch.masked_select(r, (1 - advanced.float()).byte()).view(PARAMS,-1)
        cs = torch.masked_select(r, advanced).view(PARAMS, -1)
        if not ((i < self.graph.intersections and (i+1) % self.graph.n == 0) or
            (i >= self.graph.intersections and i+self.graph.n > self.graph.roads)):
          newrd = self.graph.nexts[i]
          self.score += cs[xi,:].sum() / (self.graph.roads * 250)
          for j in range(cs.size(1)):
            cs[xi,j] = cs[xi,j] - self.graph.len
            self.add_car(newrd, cs[:,j])
  
  def update_lights(self, phase):
    # phase should be of size lights x 2
    road_phases = torch.cat((phase, phase @ self.flipv), dim=0)
    newxs = (road_phases * self.stoplightv).sum(1)
    for i in range(self.graph.roads):
      self.cars[i][xi,0] = newxs[i]

  def obs(self):
    return torch.cat([(c[xi,1:]).sum() if c.size(1) > 1
      else self.zerov for c in self.cars])


  # Think about where the gradient stops flowing
  # So here our score is increased when a car moves from one road to another.
  # Not really relevant. 

  # Can we do something easier?
  # Sure. Always have cars flowing from one direction. 
  # Could try minimizing deceleration. 

  # Even easier. Have a single road with multiple lights along it. 
  # The action can be to turn exactly one of the lights green and the rest red. 
  # The input is the x position of the first k cars. 
  # Seems easier to manage the gradient

  # Could try not doing this car moving thing.
  
  
  Each car stays on a single 
