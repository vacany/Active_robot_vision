# import sensor_msgs.point_cloud2 as pc2
import numpy as np
import rosbag
import os
import yaml



# ROSBAG SETTINGS
# data_dir = "/home/patrik/data/small_town_simulation/"
data_dir = "/home/patrik/mnt/rci/mnt/personal/vacekpa2/data/eth/station/sequence_1"
# bag_name = 'small_town_simulation.bag'
bag_name = '2020-02-20-17-28-39.bag'
bag_file = os.path.join(data_dir, bag_name)

bag = rosbag.Bag(bag_file, "r")

info_dict = yaml.load(rosbag.Bag(bag_file, 'r')._get_yaml_info(), Loader=yaml.Loader)
# Print the information contained in the info_dict
info_dict.keys()
for topic in info_dict["topics"]:
    print("-"*50)
    for k,v in topic.items():
        print(k.ljust(20), v)

from pyrosenv.sensor_msgs import point_cloud2

for frame_id, (topic, msg, t) in enumerate(bag.read_messages()):
    if 'points' in topic:
        print(frame_id, ' point cloud')

        lidar = point_cloud2.read_points(msg)
        lidar = np.array(list(lidar), dtype=np.float32)
        break



csv = open(os.path.join(data_dir, 'indices.csv'), 'r')
lines = csv.readlines()

anno_dict = {}
for line in lines:
    line = line[:-1]
    split = line.split(',')
    t = int(split[0])
    dynamic_indices = np.array(split[1:], dtype=int)
    break

dynamic_mask = np.zeros(lidar.shape[0], dtype=int)
dynamic_mask[dynamic_indices] = 1

from visualizer import visualize_points3D
visualize_points3D(lidar, dynamic_mask)
# visualize_points3D(lidar, lidar[:,3])

# unpack it on server
