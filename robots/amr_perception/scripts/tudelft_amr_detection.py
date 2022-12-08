from easy_inference.providers.realsense import Realsense
from easy_inference.providers.utils import combine
from easy_inference.utils.boundingbox import BoundingBox
from easy_inference.utils.filters import filter_iou3d

import onnxruntime as ort
import numpy as np
import cv2
import os
import time

SHOW = True
ROS = False

if ROS:
    from easy_inference.utils.ros_connector import RosConnector
    ros_connector = RosConnector(fixed_frame="map")

# ort.set_default_logger_severity(0)
models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/models'
ort_sess = ort.InferenceSession(f'{models_dir}/yolov7-tiny.onnx', providers=['CUDAExecutionProvider'])

# cam5 = Realsense(width=640, height=480, depth=True, device='215122255929')
# cam4 = Realsense(width=640, height=480, depth=True, device='215122255934')
# cam3 = Realsense(width=640, height=480, depth=True, device='215122254701')
# cam2 = Realsense(width=640, height=480, depth=True, device='114222251376')
# cam1 = Realsense(width=640, height=480, depth=True, device='215122255869')

cam1 = Realsense(width=640, height=480, depth=True, device='117222251204')
# providers = [cam1, cam2, cam3, cam4, cam5]
providers = [cam1]

import open3d as o3d
import time
fx = cam1._depth_intr.fx
fy = cam1._depth_intr.fy
cx = cam1._depth_intr.ppx
cy = cam1._depth_intr.ppy
depth_scale = cam1._depth_scale

def project_depth_to_pcl(rgb, depth):
    pcl = []
    height, width = depth.shape
    for i in range(height):
       for j in range(width):
           z = depth[i][j]

           if z < 0.1 or z > 20: continue

           r = rgb[i,j,0]
           g = rgb[i,j,1]
           b = rgb[i,j,2]
           x = (j - cx) * z / fx
           y = (i - cy) * z / fy

           pcl.append([x, y, z, r, g, b])

    pcl = np.stack(pcl)
    return pcl

import pptk

v = pptk.viewer(np.random.rand(100,3))


for frames in combine(*providers):
    rgb_frames = np.stack([f[1] for f in frames])
    depth_frames = np.stack([f[0] for f in frames])



    input = rgb_frames.transpose((0, 3, 1, 2))
    input = np.ascontiguousarray(input)
    input = input.astype(np.float32)
    input /= 255



    pcl = project_depth_to_pcl(rgb_frames[0], depth_frames[0] * depth_scale)
    #v.get() # Get the current camera config and set it for next iter


    # cv2.imshow(f"RGB", rgb_frames[0])
    # cv2.imshow(f"DEPTH", depth_frames[0])
    # cv2.waitKey(1)

    break

pptk.viewer(pcl[:,:3], pcl[:,3:])
