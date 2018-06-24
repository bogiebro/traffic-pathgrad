import visdom

class Summary:
  def __init__(self, vis, title):
    self.win = None
    self.title = title
    self.vis = vis

  def plot(self, x, y):
    if self.win:
      self.vis.line(win=self.win, X=x, Y=y, update='append')
    else:
      self.win = self.vis.line(X=x, Y=y, opts={'title': self.title})

  def hist(self, vals):
    self.win = self.vis.histogram(win=self.win, X=vals, opts={'title': self.title})
    # mean = vals.mean()
    # std = vals.std()
    # filtered = (vals > (mean - 1.9*std)) * (vals < mean + 1.9*std)
    # newvals = vals[filtered]
    # if len(newvals) > 0:
    #   self.win = vis.histogram(win=self.win, X=newvals, opts={'title': self.title})
    # elif len(vals) > 0:
    #   self.win = vis.histogram(win=self.win, X=vals, opts={'title': self.title})
