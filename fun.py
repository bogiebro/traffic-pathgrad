import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import math
from gumbel import gumbel_softmax

gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5
polemass_length = masspole * length
force_mag = 10.0
tau = 0.02

theta_threshold_radians = 10 * 2 * math.pi
x_threshold = 2

direct = Variable(torch.Tensor([-1, 1])) 
def step_cartpole(state, action):
  x, x_dot, theta, theta_dot = torch.split(state, 1, 1)
  force = ((action @ direct) * force_mag).unsqueeze(1)
  costheta = torch.cos(theta)
  sintheta = torch.sin(theta)
  addend = theta_dot * theta_dot * sintheta * polemass_length
  temp = (force + addend) / total_mass
  thetaacc = (-temp * costheta + gravity * sintheta) / (length * (4.0/3.0
    - masspole * costheta * costheta / total_mass))
  xacc = temp - polemass_length * thetaacc * costheta / total_mass
  new_x = x + tau * x_dot
  new_x_dot = x_dot + tau * xacc
  new_theta = theta + tau * theta_dot
  new_theta_dot = theta_dot + tau * thetaacc
  abs_x = new_x.abs() / x_threshold
  abs_theta = new_theta.abs() / theta_threshold_radians
  done = F.threshold(abs_x, 1, 0) + F.threshold(abs_theta, 1, 0)
  return torch.cat((new_x, new_x_dot, new_theta, new_theta_dot), 1), 1, done.squeeze(1)

w = torch.FloatTensor(4, 2)
init.xavier_normal(w)
w = Variable(w, requires_grad=True)

optimizer = optim.Adam([w], lr=1e-1)

BATCH_SIZE = 30

st = torch.FloatTensor(BATCH_SIZE,4)
dreward = torch.Tensor([-1]).expand(BATCH_SIZE)
for ep in range(1000):
  st.uniform_(-0.05, 0.05)
  vst = Variable(st)
  reward = Variable(torch.zeros(BATCH_SIZE))
  multiplier = 1
  notdone = Variable(torch.ones(BATCH_SIZE))
  for i in range(800):
    logits = vst @ w
    y = gumbel_softmax(logits, tau=1, hard=True)
    vst, r, d = step_cartpole(vst, y)
    tester = notdone * (1 - d) * r * multiplier
    reward.add_(tester)
    notdone = (d == 0).float()
    multiplier *= 0.99
    if (d.data > 0).all(): break
  if ep % 5 == 0:
    print("Num steps", i)
    optimizer.zero_grad()
  reward.backward(dreward)
  optimizer.step()
