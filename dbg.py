import rg
import idm
import bv
from torch.autograd import Variable

def stupid(i):
  pass

# idm.set_render_rate()
g = rg.GridRoad(1,1,250)
model = idm.IDM(g)
model.reset()
model.set_phase(Variable(model.random_phase()))
bv.animate(model, stupid)
