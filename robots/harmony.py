# import sensor_msgs.point_cloud2 as pc2
import numpy as np
import rosbag
import os
import yaml



# ROSBAG SETTINGS

data_dir = "/home/patrik/data/"
bag_name = 'Rosbag_Nicky_2022_09_30.bag'
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


topics = bag.get_type_and_topic_info()[1].keys()
print(topics)
types = []


for frame_id, (topic, msg, t) in enumerate(bag.read_messages()):
    if '/kinect_master/depth/image_raw/compressed' in topic:
        breakpoint()
    # pass
    # if 'points' in topic:
        # print(frame_id, ' point cloud')

        # lidar = point_cloud2.read_points(msg)
        # lidar = np.array(list(lidar), dtype=np.float32)


# visualize_points3D(lidar, lidar[:,3])

# unpack it on server
