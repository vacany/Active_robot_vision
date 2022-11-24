import os.path
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import yaml

import visualizer
# rewrite to dataset class, loadable from main dataset class

fx= 540.686
fy = 540.686

cx = 479.75
cy = 269.75

HOSPITAL_PATH = os.path.expanduser("~") + '/data/hospital/'

testset2_info = open(HOSPITAL_PATH + '/ImageSets/TestSet2_seq1.txt')
frame = testset2_info.readlines()[500][:-1]

image_path = HOSPITAL_PATH + f'/Images_RGB/{frame}.png'
depth_path = HOSPITAL_PATH + f'/Images_Depth/{frame}.png'
annotation_path = HOSPITAL_PATH + f'/Annotations_RGB_TestSet2/{frame}.yml'


# equipped with a Kinect v2 camera mounted 1 m above
#the ground and capturing image


rgb = plt.imread(image_path)
depth = plt.imread(depth_path)

depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
depth = depth / 1000     # scalling factor - taken from https://github.com/marinaKollmitz/hospital_people_detector/blob/master/src/find_depth.cpp

plt.imshow(rgb)
plt.show()
plt.imshow(depth)
plt.show()

with open(annotation_path, 'r') as f:
    anno = yaml.load(f, Loader=yaml.Loader)

boxes = anno['annotation']['object']
box_list = []

seg_image = np.zeros(depth.shape)
depth_limits = (0.5,0.5)    # annotation in center, What to take inside the box (before center, after center)

for box in boxes:
    clz = box['name']
    dims = box['bndbox']
    d = float(dims['depth'])

    # Not in the image it seems
    if d == -1:
        continue

    xmax = int(dims['xmax'])
    xmin = int(dims['xmin'])
    ymax = int(dims['ymax'])
    ymin = int(dims['ymin'])

    area_mask = np.zeros(seg_image.shape, dtype=bool)
    area_mask[ymin : ymax, xmin : xmax] = True
    depth_mask = (depth < d + depth_limits[1]) & (depth > d - depth_limits[0])

    seg_image[area_mask & depth_mask] = 1

# Projection, eliminate noise
pcl = []
height, width = depth.shape
for i in range(height):
   for j in range(width):
       z = depth[i][j]

       if z < 0.1 or z > 10: continue

       r = rgb[i,j,0]
       g = rgb[i,j,1]
       b = rgb[i,j,2]
       x = (j - cx) * z / fx
       y = (i - cy) * z / fy

       segmentation = seg_image[i,j]

       pcl.append([x, y, z, r, g, b, segmentation])

pcl = np.stack(pcl)

import pptk
v=pptk.viewer(pcl[:,:3], pcl[:,3:6])
v.set(point_size=0.003)
pptk.viewer(pcl[:,:3], pcl[:,-1])

num_points = 40000
inds = np.random.choice(pcl.shape[0], num_points, replace=False)
v2 = pptk.viewer(pcl[inds,:3], pcl[inds,3:6])
v2.set(point_size=0.003)
