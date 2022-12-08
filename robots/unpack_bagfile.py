# import sensor_msgs.point_cloud2 as pc2
import glob
import shutil

import numpy as np
import rosbag
import os
import yaml
import open3d as o3d
from pyrosenv.sensor_msgs import point_cloud2
import subprocess

import sys
import rospy
from PIL import Image

# from cv_bridge import CvBridge

def run_in_shell(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)


def load_pcd_to_npy(file):

    pcd = o3d.io.read_point_cloud(file)
    out_arr = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    pcl_rgb = np.concatenate((out_arr, colors), axis=1)

    return pcl_rgb


def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv




def unpack_bag(bag_file, data_dir):

    bag = rosbag.Bag(bag_file, "r")


    info_dict = yaml.load(rosbag.Bag(bag_file, 'r')._get_yaml_info(), Loader=yaml.Loader)
    # Print the information contained in the info_dict
    info_dict.keys()
    for topic in info_dict["topics"]:
        print("-" * 50)
        for k, v in topic.items():
            print(k.ljust(20), v)

    # THIS CAN THROW ERROR, SHOULD BE RUN MANUALLY IN THAT CASE
    run_in_shell([f'rosrun pcl_ros bag_to_pcd {bag_file} /camera/depth/color/points {data_dir}/realsense_pcl/'])

    for folder in ['raw_depth', 'raw_rgb', 'raw_realsense_pcl', 'raw_velodyne', 'raw_camera_pts']:
        os.makedirs(data_dir + '/' + folder, exist_ok=True)

    for topic, msg, t in bag.read_messages():

        if 'velodyne_points' in topic:
            lidar = point_cloud2.read_points(msg)
            lidar = np.array(list(lidar), dtype=np.float32)

            np.save(f'{data_dir}/raw_velodyne/{t}.npy', lidar)

        if 'tf' in topic:
            pass

        if 'camera/depth/color/points' in topic:
            print(topic, 'implemented outside the python')
            pass

        if 'camera/depth/image_rect_raw' in topic:
            depth = np.ndarray(shape=(msg.height, msg.width), dtype=np.dtype('uint16'), buffer=msg.data)
            img = Image.fromarray(depth)
            img.save(f'{data_dir}/raw_depth/{t}.png')

        if 'camera/color/image_raw' in topic:

            # color_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            color_img = imgmsg_to_cv2(msg)
            img = Image.fromarray(color_img)
            img.save(f'{data_dir}/raw_rgb/{t}.png')
            # resize it here?


    for realsense_pcd in sorted(glob.glob(f'{data_dir}/realsense_pcl/*.pcd')):
        pcl = load_pcd_to_npy(realsense_pcd)

        pcl = pcl[:, [2, 0, 1, 3, 4, 5]]
        pcl[:, 2] = - pcl[:, 2]
        pcl[:, 1] = - pcl[:, 1]
        # do it over the radius
        pcl_rgb = pcl[(pcl[:, 0] < 10) & (pcl[:, 0] > 0.3)]

        new_name = os.path.dirname(realsense_pcd) + '/../raw_camera_pts/'
        ts = "".join(os.path.basename(realsense_pcd).split('.')[:2])

        np.save(new_name + '/' + ts + '.npy', pcl_rgb)

    shutil.rmtree(data_dir + '/realsense_pcl')

if __name__ == "__main__":
    import sys
    unpack_bag(sys.argv[1], sys.argv[2])
