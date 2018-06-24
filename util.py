import flags
import rg
import idm
from itertools import count

def main(pol):
  args = flags.parser.parse_args()
  flags.derive_opts(args)
  g = rg.GridRoad(3, 3, 250)
  model = idm.IDM(g)
  model.reset()
  p = pol(args, model)
  if args.render:
    import bv
    bv.animate(model, p.policy)
  else:
    for i in count(0):
      p.policy(i)
      model.step()

