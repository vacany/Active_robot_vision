import numpy as np

import visualizer
import matplotlib.pyplot as plt
from data_utils.delft_dataset import Delft_Sequence

from data_utils.instances import Object3D
from timespace.trajectory import Trajectory

Jackal = {'x' : 0.,
          'y' : 0.,
          'z' : 0.,
          'l' : 0.5,
          'w' : 0.4,
          'h' : 0.4,
          'yaw' : 0.}




# init object

Sequence = Delft_Sequence()
pts = Sequence.get_feature(0, name='camera_pts')
# Mask distance
pts = pts[(pts[:,0] ** 2 < 64) & (pts[:,1] ** 2 < 64)]



plt.plot(pts[:,0], pts[:,1], 'b.', markersize=0.1)

# To get the odometry by clicking
robot_odometry = np.array(plt.ginput(15))
np.save('robots/camera_odometry.npy', robot_odometry)
# click it in both

robot_odometry = np.load('robots/camera_odometry.npy')

# todo ground not solved yet
robot_z = np.full_like(robot_odometry[:,0], fill_value=-0.5)
robot_t = np.arange(0, len(robot_z), len(robot_z) + 1)

traj = Trajectory(robot_odometry[:, 0], robot_odometry[:, 1], zs=robot_z, ts=robot_t)
traj.assign_body(Jackal['l'], Jackal['w'], Jackal['h'])
traj.plot()
robot_pts = traj.boxes_points
# Todo time pts should be done better

robot = Object3D(pts=robot_pts[:, :3], time_pts=np.ones(robot_pts.shape[0]), name='robot')


visualizer.visualize_multiple_pcls(pts, robot.pts)

# try the
