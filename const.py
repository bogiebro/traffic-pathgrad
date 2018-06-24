import util
from summaries import Summary
from rg import mc
import torch
from torch.autograd import Variable

class ConstPolicy:
  def __init__(self, opts, model):
    self.opts = opts
    self.model = model
    self.scores = []
    self.score_summary = Summary(opts.vis, "Const Score")
    greens = Variable(mc(torch.ones(model.graph.intersections)))
    if opts.render: greens.volatile = True
    # self.phase = torch.stack((greens, 1 - greens), 1)
    self.phase = torch.stack((1 - greens, greens), 1)
    self.episode = 0

  def policy(self, i):
    if (i % self.opts.episode_ticks) == (-1 % self.opts.episode_ticks):
      self.scores.append(self.model.score().data)
      if self.episode % self.opts.refresh_rate == 0 and len(self.scores) > 1:
        self.score_summary.hist(torch.cat(self.scores, 0))
      self.episode += 1
    if i % self.opts.episode_ticks == 0:
      self.model.reset()
    if i % self.opts.light_ticks == 0:
      self.model.set_phase(self.phase)

if __name__ == '__main__':
  util.main(ConstPolicy)
