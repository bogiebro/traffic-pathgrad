import argparse
import idm

def derive_opts(opts):
  if opts.render: idm.set_render_rate()
  opts.episode_lights = int(opts.episode_secs / opts.light_secs)
  opts.light_ticks = int(opts.light_secs / idm.RATE)
  opts.episode_ticks = opts.episode_lights * opts.light_ticks
  opts.obs_ticks = int(opts.obs_secs / idm.RATE)
  if opts.visdom:
    import visdom
    opts.vis = visdom.Visdom(env=opts.visdom)

parser = argparse.ArgumentParser()
parser.add_argument('--episode_secs', type=int, default=100)
parser.add_argument('--light_secs', type=int, default=5)
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--save_rate', type=int, default=100)
parser.add_argument('--summary_rate', type=int, default=10)
parser.add_argument('--restore', type=bool, default=False)
parser.add_argument('--spacing', type=int, default=3)
parser.add_argument('--refresh_rate', type=int, default=10)
parser.add_argument('--step_rate', type=int, default=5)
parser.add_argument('--obs_secs', type=int, default=1)
parser.add_argument('--visdom', default=None)
parser.add_argument('--render', type=bool, default=False)

