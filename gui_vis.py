import time

import numpy as np
import os.path
from visualizer import *

former_pts = 0
former_color = 0

v = visualize_points3D(np.random.rand(100,3))
while True:

  pts_path = '/home/patrik/data/tmp_vis/visul.npy'
  color_path = '/home/patrik/data/tmp_vis/visul_color.npy'

  if os.path.exists(pts_path):
    pts = np.load(pts_path)

    if os.path.exists(color_path):
      color = np.load(color_path, allow_pickle=True)

    if np.array_equal(former_pts, pts) and np.array_equal(former_color, color):
      print("No different between last")
      time.sleep(1)
    else:
        # We got new visuals!
      v.load(pts[:,:3], color)
      former_pts = pts
      former_color = color

      time.sleep(1)
