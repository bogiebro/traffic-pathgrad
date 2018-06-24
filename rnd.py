from torch.autograd import Variable
import idm

class RandPol:
  def __init__(self, args, model):
    self.model = model
    self.args = args
  def policy(self, i):
    if i % self.args.light_ticks == 0:
      self.model.set_phase(Variable(self.model.random_phase(), volatile=True))

if __name__ == '__main__':
  import util
  util.main(RandPol)
