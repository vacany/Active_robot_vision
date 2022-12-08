import matplotlib
import os.path
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import yaml
from PIL import Image

import visualizer


def project_depth_to_pcl(depth, rgb, seg_image=None):
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

           if seg_image is not None:
            seg_r = seg_image[i,j,0] / 255
            seg_g = seg_image[i,j,1] / 255
            seg_b = seg_image[i,j,2] / 255


            pcl.append([x, y, z, r, g, b, seg_r, seg_g, seg_b])
           else:
            pcl.append([x, y, z, r, g, b])

    pcl = np.stack(pcl)
    return pcl

def plot_bev(pcl, save=None, lim=None):
    fig, ax = plt.subplots()
    ax.scatter(pcl[:, 0], pcl[:, 1], color=pcl[:, 6:9], marker='.', s=0.2)
    ax.set_xlim([-.5, 6])
    ax.set_ylim([-4, 4])

    if save is not None:
        fig.savefig(top_down_inference)
    else:
        plt.show()

    plt.close()

def plot_realsense(pcl, features, save=None):
    fig = plt.figure(figsize=(5, 5), dpi=200)
    ax = fig.add_subplot(projection='3d')

    xs = pcl[:, 0]
    ys = pcl[:, 1]
    zs = pcl[:, 2]

    ax.scatter(xs, ys, zs, marker='.', s=0.3, c=features, alpha=0.6)
    ax.set_xlim([-.5, 6])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    ax.view_init(elev=25, azim=210)
    ax.dist = 6
    plt.axis('off')

    if save is not None:
        plt.savefig(pcl_inference)
    else:
        plt.show()

    plt.close()


# Hospital dataset
# fx= 540.686
# fy = 540.686

# cx = 479.75
# cy = 269.75
#
# Realsense

# HOSPITAL_PATH = os.path.expanduser("~") + '/data/hospital/'

    # with open(annotation_path, 'r') as f:
    #     anno = yaml.load(f, Loader=yaml.Loader)

    # boxes = anno['annotation']['object']
    # box_list = []

    # seg_image = np.zeros(depth.shape)
    # depth_limits = (0.5,0.5)    # annotation in center, What to take inside the box (before center, after center)
    #
    # for box in boxes:
    #     clz = box['name']
    #     dims = box['bndbox']
    #     d = float(dims['depth'])
    #
    #     Not in the image it seems
        # if d == -1:
        #     continue
        #
        # xmax = int(dims['xmax'])
        # xmin = int(dims['xmin'])
        # ymax = int(dims['ymax'])
        # ymin = int(dims['ymin'])
        #
        # area_mask = np.zeros(seg_image.shape, dtype=bool)
        # area_mask[ymin : ymax, xmin : xmax] = True
        # depth_mask = (depth < d + depth_limits[1]) & (depth > d - depth_limits[0])
        #
        # seg_image[area_mask & depth_mask] = 1
    #
    # Projection, eliminate noise
# testset2_info = open(HOSPITAL_PATH + '/ImageSets/TestSet2_seq1.txt')
# frame = testset2_info.readlines()[500][:-1]

# image_path = HOSPITAL_PATH + f'/Images_RGB/{frame}.png'
# depth_path = HOSPITAL_PATH + f'/Images_Depth/{frame}.png'
# annotation_path = HOSPITAL_PATH + f'/Annotations_RGB_TestSet2/{frame}.yml'


# equipped with a Kinect v2 camera mounted 1 m above
#the ground and capturing image


# Realsense
# directly from realsense camera pyrealsense2
realsense_c = [421.827, 239.454]
realsense_f = [427.01, 427.01]
cx, cy = realsense_c[0], realsense_c[1]
fx, fy = realsense_f[0], realsense_f[1]

DATASET_PATH = os.path.expanduser("~") + '/patrik_data/delft_toy/'

rgb_paths = sorted(glob.glob(os.path.expanduser("~") + '/data/robots/raw_rgb/*.png'))
depth_paths = sorted(glob.glob(os.path.expanduser("~") + '/data/robots/raw_depth/*.png'))
seg_paths = sorted(glob.glob(os.path.expanduser("~") + '/data/robots/hrnet_seg/*.png'))

for frame in range(len(rgb_paths)):
    print(frame)

    depth = cv2.imread(depth_paths[frame], cv2.IMREAD_UNCHANGED)
    rgb = np.asarray(Image.open(rgb_paths[frame]).resize((depth.shape[1], depth.shape[0]))) / 255
    seg_img = np.asarray(Image.open(seg_paths[frame]))
    resized_seg_img = cv2.resize(seg_img, (rgb.shape[1], rgb.shape[0]))

      # scalling factor - taken from https://github.com/marinaKollmitz/hospital_people_detector/blob/master/src/find_depth.cpp

    for i in range(20):
        depth = cv2.medianBlur(depth, ksize=5)

    depth = depth / 1000

    pcl = project_depth_to_pcl(depth, rgb, seg_image=resized_seg_img)

    pcl[:,[0,1,2]] = pcl[:,[2,0,1]]
    pcl[:, 2] = - pcl[:, 2]
    pcl[:, 1] = - pcl[:, 1]
    # do it over the radius
    pcl = pcl[(pcl[:, 0] < 10) & (pcl[:, 0] > 0.3)]

    pcl_inference = os.path.dirname(rgb_paths[frame]) + '/../hrnet_pcl_img/' + os.path.basename(rgb_paths[frame])
    pcl_colors = os.path.dirname(rgb_paths[frame]) + '/../camera_pts_color_img/' + os.path.basename(rgb_paths[frame])
    top_down_inference = os.path.dirname(rgb_paths[frame]) + '/../hrnet_filtered_img/' + os.path.basename(rgb_paths[frame])

    # PLOTING
    plot_bev(pcl, save=top_down_inference)

    break
    # plot_realsense(pcl, features=pcl[:,6:9])

