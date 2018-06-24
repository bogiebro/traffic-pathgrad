import util
from gumbel import gumbel_softmax
import idm
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from rg import mc

class IDMTrainer(nn.Module):
  def __init__(self, model, opts):
    super().__init__()
    self.model = model
    self.fc1 = nn.Linear(model.graph.roads, 50)
    self.fc2 = nn.Linear(50, 2 * model.graph.intersections)
  def forward(self, x, temp=1):
    logits = self.fc2(F.relu(self.fc1(x))).view(-1, 2)
    result = gumbel_softmax(logits, temp, hard=True)
    return logits, result

def to_temp(i):
  return 1
  # return max(10 - (i / 50), 1)

BATCH_SIZE = 10

class TrainedPolicy:
  def __init__(self, opts, model):
    self.dreward = torch.Tensor([-1])
    self.net = IDMTrainer(model, opts)
    self.model = model
    self.opts = opts
    if opts.restore: net.load_state_dict(torch.load("checkpoint"))
    mc(self.net)

    if opts.visdom:
      from summaries import Summary
      self.avg_score = Summary(opts.vis, "Score")
      self.grads = Summary(opts.vis, "Grads")
    self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
    self.optimizer.zero_grad()

  def policy(self, i):
    episode = i // self.opts.episode_ticks
    k = i // self.opts.light_ticks
    if (i % self.opts.episode_ticks) == (-1 % self.opts.episode_ticks):
      print("Score is", self.model.score.data[0])
      self.model.score.backward(self.dreward)
      if episode % self.opts.summary_rate == 0:
        if self.opts.visdom:
          gs = [p.grad.data.cpu().view(-1) for p in self.net.parameters()]
          self.grads.hist(torch.cat(gs,0))
          self.avg_score.plot(torch.Tensor([episode]), self.model.score.data.cpu())
      if episode % self.opts.step_rate == 0:
        self.optimizer.step()
        self.optimizer.zero_grad()
    if i % self.opts.episode_ticks == 0:
      self.model.reset()
    if i % self.opts.light_ticks == 0:
      pos = self.model.obs().unsqueeze(0)
      logits, action = self.net(pos, temp=to_temp(episode))
      self.model.set_phase(action)

if __name__ == '__main__':
  util.main(TrainedPolicy)
