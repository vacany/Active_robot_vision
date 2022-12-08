import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import torch.utils.data
import yaml

from PIL import Image


SSHFS_ROOT = '/home/patrik/mnt/rci/'
DATA_ROOT = '/mnt/personal/vacekpa2/data/'

SEMANTIC_KITTI_ROOT = '/semantic_kitti/dataset/sequences/'
SYNLIDAR_ROOT = '/synlidar/SubDataset/sequences/'
ARGOVERSE2_TRAIN_ROOT = '/argoverse2/sensor/train/'
JACKAL_ROOT = 'Not implemented yet'

import socket
if socket.gethostname() == ('Patrik'):
    pass # leave out for now
    DATA_ROOT = '/home/patrik/patrik_data/'
    # DATA_ROOT = SSHFS_ROOT + DATA_ROOT

PATH_DICT = {'semantic_kitti': DATA_ROOT + SEMANTIC_KITTI_ROOT,
             'synlidar' : DATA_ROOT + SYNLIDAR_ROOT,
             'argoverse2_train': DATA_ROOT + ARGOVERSE2_TRAIN_ROOT}


class Basic_Dataprocessor(object):

    def __init__(self, data_dir):
        self.data_dir = data_dir


    def unified_preload(self, path : str):
        if path.endswith('.bin'):
            return np.fromfile(path, dtype=np.float32).reshape(-1,4)

        elif path.endswith('.npy'):
            return np.load(path, allow_pickle=True)

        elif path.endswith('.png') or path.endswith('.jpg'):
            return np.asarray(Image.open(path))

        elif path.endswith('.label'):
            return NotImplementedError("SemanticKitti format")
        else:
            raise NotImplementedError("Different formats")

    def unified_store(self, feature, path:str):
        if path.endswith('.bin'):
            return np.fromfile(path, dtype=np.float32).reshape(-1,4)

        elif path.endswith('.npy'):
            return np.load(path, allow_pickle=True)

        elif path.endswith('.png') or path.endswith('.jpg'):
            return np.asarray(Image.open(path))

    @classmethod
    def pts_to_frame(self, pts, pose):
        '''

        :param pts: point cloud
        :param pose: 4x4 transformation matrix
        :return:
        '''
        transformed_pts = pts.copy()
        transformed_pts[:, :3] = (np.insert(pts[:, :3], obj=3, values=1, axis=1) @ pose.T)[:, :3]

        return transformed_pts

    @classmethod
    def frame_to_pts(self, pts, pose):
        raise NotImplementedError("Not yet needed")

    def get_global_pts(self, idx, name):
        pts = self.get_feature(idx, name)
        pose = self.get_feature(idx, 'pose')
        global_pts = self.pts_to_frame(pts, pose)

        return global_pts

    def get_feature(self, idx, name : str):
        path_to_feature = glob.glob(self.data_dir + name + f'/{idx:06d}.*')[0]

        return self.unified_preload(path_to_feature)

    def store_feature(self, feature, idx, name):

        path_to_feature = self.data_dir + name + f'/{idx:06d}.npy'

        os.makedirs(os.path.dirname(path_to_feature), exist_ok=True)

        np.save(path_to_feature, feature)

    def get_range_of_feature(self, start, end, name):
        features = [self.get_feature(i, name) for i in range(start, end)]
        return features






if __name__ == '__main__':
    pass
