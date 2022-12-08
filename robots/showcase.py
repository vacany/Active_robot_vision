import matplotlib.pyplot as plt
import numpy as np
import glob
import os

import open3d as o3d
from unpack_bagfile import load_pcd_to_npy

from data_utils import bev
from visualizer import *


DATA_DIR = '/home/patrik/patrik_data/'

if __name__ == '__main__':
    SENSOR = 'realsense'

    if SENSOR == 'realsense':
        data = sorted(glob.glob(DATA_DIR + SENSOR + '/out_pcd/*.pcd'))
    elif SENSOR == 'velodyne':
        data = sorted(glob.glob(DATA_DIR + SENSOR + '/out_npy/*.npy'))
    else:
        raise NotImplementedError("Supported sensors are only velodyne and realsense")

    load_data = np.load if SENSOR == 'velodyne' else load_pcd_to_npy

    for sample in data:
        pcl = load_data(sample)
        break

    Bev_template = bev.BEV()
    Bev_template.create_bev_template_from_points(pcl)
    grid = Bev_template.generate_bev(pcl, pcl[:,3])



    # click odometry is reverse order when using ginput
    clicked_odometry = np.array(((9.840368, 29.357143),
                                 (16.13906, 30.406926),
                                 (19.813312, 34.343615)), dtype=float) / 10
    # division by 10 is image_grid ratio
    for position in clicked_odometry:
        coors = (position * 10).astype(int)
        grid[coors[0], coors[1]] = 1

    plt.imshow(grid)
    plt.show()

    if SENSOR == 'velodyne':
        visualize_points3D(pcl, pcl[:,3])

    elif SENSOR == 'realsense':
        visualize_points3D(pcl, pcl[:,3:])

