import os.path
import glob
import socket
from shutil import copy2
import numpy as np

import visualizer
from data_utils.basics import Basic_Dataprocessor
from timespace.timestamps import find_nearest_timestamps

if socket.gethostname().startswith('Patrik'):
    DATASET_PATH = os.path.expanduser("~") + '/patrik_data/delft_toy/'
else:
    DATASET_PATH = os.path.expanduser("~") + '/data/robots/'

class Delft_Sequence(Basic_Dataprocessor):

    def __init__(self, sequence=0):

        Basic_Dataprocessor.__init__(self, data_dir=DATASET_PATH)

        self.sequence = sequence
        self.raw_features = ['raw_camera_pts', 'raw_depth', 'raw_rgb', 'raw_velodyne', 'raw_poses']
        self.useable_features = ['camera_pts', 'depth', 'rgb', 'velodyne', 'poses']

        self.get_available_data()
        self.synchronize_by_feature(feature_name='raw_velodyne')    # synchronization to lowest frame rate

    def get_available_data(self):
        '''
        Scans through main folder. It assumes that all data are one level bellow the root
        :return:
        '''
        self.data_paths = {}

        for key in self.raw_features:
            self.data_paths[key] = sorted(glob.glob(self.data_dir + '/' + key + '/*'))


    def synchronize_by_feature(self, feature_name):
        '''
        #todo in future to the basic dataprocessor
        :param feature_name: Choose one raw feature and find the closest nearby frames from other features
        :return: list of the frames with all file_paths without RAW PREFFIX!
        '''
        self.data_frames = []

        frames = self.data_paths[feature_name]

        for frame in frames:
            data_dict = {}
            timestamp = int(os.path.basename(frame).split('.')[0])

            for corres_feature in self.raw_features:
                all_timestamps = [int(os.path.basename(feat_f).split('.')[0]) for feat_f in self.data_paths[corres_feature]]

                index = find_nearest_timestamps(all_timestamps, timestamp)
                data_dict[corres_feature.replace('raw_', '')] = self.data_paths[corres_feature][index]

            self.data_frames.append(data_dict)

        return self.data_frames

    def store_by_data_frames(self):
        '''

        :return: Store the data into same folder
        '''
        for idx, frame_dict in enumerate(self.data_frames):

            for key, value in frame_dict.items():
                feature = self.unified_preload(value)
                self.store_feature(feature, idx, name=key)


    def __getitem__(self, idx):
        data = {}
        for key, value in self.data_frames[idx].items():
            data[key] = self.unified_preload(value)

        return data

    def __len__(self):
        return len(self.data_frames)



if __name__ == '__main__':
    sequence = Delft_Sequence()
    # sequence.store_by_data_frames()

    robot_odometry = np.load(
        '../robots/camera_odometry.npy')  # transfer poses to format so that I can load it uniformly
    robot_z = np.full_like(robot_odometry[:, 0], fill_value=-0.5)
    robot_odometry = np.stack((robot_odometry[:, 0], robot_odometry[:, 1], robot_z)).T

    # you are missing scripts for running, regeneration etc. the final products
