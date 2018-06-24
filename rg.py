import numpy as np
import torch

def mc(a):
  return a
  # try: return a.cuda()
  # except: return a

class GridRoad:
  def __init__(self, m, n, l):
    v = m*n
    self.to_roads = torch.cat((torch.Tensor([[1, 0]]).expand(v, 2),
      torch.Tensor([[0, 1]]).expand(v, 2)), dim=0)
    self.len = l
    self.m = m
    self.n = n
    self.intersections = v
    self.roads = 2*v
    self.nexts = np.concatenate((np.arange(v) + 1, n + np.arange(v)))
    self.locs = l * get_locs_gridroad(m,n,v,self.roads)
    self.entrypoints = np.concatenate((v + np.arange(n), (n * np.arange(m)))
        ).astype(np.int32)

def get_locs_gridroad(m,n,v,roads):
  locs = np.empty((roads,2,2), dtype=np.float32)
  for i in range(roads):
    d = i // v
    li = i % v
    col = li % n
    row = li // n
    if d == 0: locs[i] = np.array(((col-1,row),(col,row)))
    else: locs[i] = np.array(((col,row+1),(col,row)))
  return locs
