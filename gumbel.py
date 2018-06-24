import torch
from torch.autograd import Variable
import torch.nn.functional as F

def sample_gumbel(a):
  noise = torch.rand(a)
  try: noise = noise.cuda()
  except: pass
  eps = 1e-10
  noise.add_(eps).log_().neg_()
  noise.add_(eps).log_().neg_()
  return Variable(noise)

def gumbel_softmax(a, tau=1, hard=True):
  shape = a.size()
  noise = sample_gumbel(shape)
  y_soft = F.softmax((a + noise) / tau, dim=a.dim()-1)
  if hard:
    _, k = y_soft.data.max(-1)
    y_hard = a.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
    return Variable(y_hard - y_soft.data) + y_soft
  else: return y_soft

if __name__ == '__main__':
  logits = Variable(torch.randn(1,2), requires_grad=True)
  a = gumbel_softmax(logits, 1, hard=False)
  a.backward(torch.ones(a.size()))
  print("Logits", logits.data)
  print("a", a.data)
  print("Grad", logits.grad)

