from vispy import app, scene
import numpy as np
import rg
import idm
import torch
from torch.autograd import Variable
import flags
from rg import mc

nocars = np.zeros((0,2),dtype=np.float32)

class Drawer:
  def __init__(self, model, fn):
    self.model = model
    self.fn = fn
    self.roadcolors = np.empty((model.graph.roads * 2, 4), dtype=np.float32)
    for i in range(model.graph.roads): self.roadcolors[2*i:2*i+2] = np.array([0,1,0,0.4])
    self.roadrots = np.zeros((model.graph.roads, 2, 2))
    roadlines = np.concatenate(model.graph.locs)
    self.canvas = scene.SceneCanvas(show=True, size=(2500, 2500))
    view = self.canvas.central_widget.add_view()
    view.camera = 'panzoom'
    view.camera.aspect = 1
    self.lines = scene.Line(pos=roadlines, connect='segments', method='gl',
      parent=view.scene, color=self.roadcolors)
    view.camera.set_range()
    self.markers = scene.Markers(parent=view.scene)
    for i in range(model.graph.roads):
      cos = model.graph.locs[i,1,0] - model.graph.locs[i,0,0]
      sin = model.graph.locs[i,1,1] - model.graph.locs[i,0,1]
      self.roadrots[i] = np.array([[cos, sin], [-sin, cos]]) / model.graph.len

  def draw(self, ev):
    self.fn(ev.iteration)
    self.model.step()
    allvals = []
    for i,a in enumerate(self.model.cars):
      if a.size(1) > 1:
        coords = np.column_stack((a.data[idm.xi,1:].cpu().numpy(), np.zeros(a.size(1)-1)))
        allvals.append(coords.dot(self.roadrots[i]) + self.model.graph.locs[i,0])
      v = i % self.model.graph.intersections
      d = i // self.model.graph.intersections
      foo = int(self.model.previous_phase.data[v,0])
      if d ^ foo:
        self.roadcolors[2*i:2*i+2] = np.array([1,0,0,1])
      else:
        self.roadcolors[2*i:2*i+2] = np.array([0,1,0,0.4])
    self.markers.set_data(np.concatenate(allvals) if allvals else nocars)
    self.lines.set_data(color=self.roadcolors)

def animate(model, fn):
  app.use_app(backend_name="Glfw")
  drawer = Drawer(model, fn)
  timer = app.Timer(interval=idm.RATE, connect=drawer.draw, start=True)
  app.run()
