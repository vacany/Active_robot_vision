import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import torch.utils.data
import yaml

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]

SSHFS_ROOT = '/home/patrik/mnt/rci/'
DATA_ROOT = '/mnt/personal/vacekpa2/data/'

SEMANTIC_KITTI_ROOT = '/semantic_kitti/dataset/sequences/'
SYNLIDAR_ROOT = '/synlidar/SubDataset/sequences/'
ARGOVERSE2_TRAIN_ROOT = '/argoverse2/sensor/train/'
JACKAL_ROOT = 'Not implemented yet'

import socket
if socket.gethostname() == ('Patrik'):
    pass # leave out for now
    DATA_ROOT = '/home/patrik/data/'
    # DATA_ROOT = SSHFS_ROOT + DATA_ROOT

PATH_DICT = {'semantic_kitti': DATA_ROOT + SEMANTIC_KITTI_ROOT,
             'synlidar' : DATA_ROOT + SYNLIDAR_ROOT,
             'argoverse2_train': DATA_ROOT + ARGOVERSE2_TRAIN_ROOT}


class Sequence_Loader(torch.utils.data.Dataset):

    def __init__(self, dataset_name='semantic_kitti', sequence=0):

        self.name = dataset_name
        if dataset_name in ['semantic_kitti', 'synlidar']:
            from pat.semantic_kitti import get_ego_bbox
            self.ego_box = get_ego_bbox()
            self.pts_files = sorted(glob.glob(PATH_DICT[dataset_name] + f"{sequence:02d}/velodyne/*.bin"))
            self.label_files = sorted(glob.glob(PATH_DICT[dataset_name] + f"{sequence:02d}/labels/*.label"))

            if dataset_name in ['semantic_kitti']:

                self.poses = np.load(PATH_DICT[dataset_name] + f'{sequence:02d}/sequence_poses.npy')

                reverse_label_name_mapping = {}
                self.label_map = np.zeros(260)
                cnt = 0

                for label_id in label_name_mapping:
                    if label_id > 250:
                        if label_name_mapping[label_id].replace('moving-',
                                                                '') in kept_labels:
                            self.label_map[label_id] = reverse_label_name_mapping[
                                label_name_mapping[label_id].replace('moving-', '')]
                        else:
                            self.label_map[label_id] = 255
                    elif label_id == 0:
                        self.label_map[label_id] = 255
                    else:
                        if label_name_mapping[label_id] in kept_labels:
                            self.label_map[label_id] = cnt
                            reverse_label_name_mapping[
                                label_name_mapping[label_id]] = cnt
                            cnt += 1
                        else:
                            self.label_map[label_id] = 255

                self.reverse_label_name_mapping = reverse_label_name_mapping
                self.num_classes = cnt

            elif dataset_name in ['synlidar']:
                with open(PATH_DICT[dataset_name] + '/../annotations.yaml', 'r') as f:
                    config_dict = yaml.load(f, yaml.Loader)['map_2_semantickitti']
                    self.label_map = np.array(list(config_dict.values())).astype(np.int32)

                self.poses = np.zeros(len(self.pts_files))

        elif dataset_name in ['argoverse2_train']:
            # todo For all argoverse, should be in argowrapper I guess.
            from pat.argoverse2 import get_ego_bbox_argo
            self.ego_box = get_ego_bbox_argo()

            log_ids = sorted(glob.glob(PATH_DICT[dataset_name] + '*'))
            log_nbr = {i: seq for i, seq in zip(range(len(log_ids)), log_ids)}

            sequence = os.path.basename(log_nbr[sequence])

            max_idx = len(glob.glob(PATH_DICT[dataset_name] + f"/{sequence}/sensors/lidar/*.feather")) - 1 # to include zero
            last_lidar_path = PATH_DICT[dataset_name] + f'/{sequence}/lidar/{max_idx:06d}.npy'

            if not os.path.exists(last_lidar_path):
                # Maybe split to different init functions...
                from pat.argoverse2 import prepare_sequence
                prepare_sequence(av2_seq_path=PATH_DICT[dataset_name] + f"{sequence}")


            self.pts_files = sorted(glob.glob(PATH_DICT[dataset_name] + f"{sequence}/lidar/*.npy"))
            self.label_files = sorted(glob.glob(PATH_DICT[dataset_name] + f"{sequence}/seg_labels/*.npy"))
            self.poses = np.load(PATH_DICT[dataset_name] + f'{sequence}/poses.npy')
            self.instance_files = sorted(glob.glob(PATH_DICT[dataset_name] + f"{sequence}/id_mask/*.npy"))

        else:
            raise NotImplementedError(f"Dataset '{dataset_name}' not implemented yet, choose between: {PATH_DICT}")



    def __len__(self):
        return len(self.pts_files)

    def __getitem__(self, index):

        pts = self.unified_preload(self.pts_files[index])
        global_pts = pts.copy()
        global_pts[:, :3] = (np.insert(pts[:, :3], obj=3, values=1, axis=1) @ self.poses[index].T)[:, :3]

        # todo refak
        if self.name in ['semantic_kitti']:
            if os.path.exists(self.label_files[index]) and len(self.label_files) > 0:
                with open(self.label_files[index], 'rb') as a:
                    all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
                    instance = all_labels >> 16  # instance id in upper half
            else:
                all_labels = np.zeros(pts.shape[0]).astype(np.int32)
                instance = np.zeros(pts.shape[0]).astype(np.int32)

            labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)


        elif self.name in ['synlidar']:
            with open(self.label_files[index], 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.int32).reshape(-1)
                instance = all_labels >> 16  # instance id in upper half

            labels_ = self.label_map[all_labels].astype(np.int64)
            global_pts = pts

        elif self.name in ['argoverse2_train']:
            labels_ = np.load(self.label_files[index])
            instance = np.load(self.instance_files[index])


        return {
            'pts': pts,
            'global_pts' : global_pts,
            # 'orig_label': all_labels,
            'pose' : self.poses[index],
            'label_mapped': labels_,
            'instance' : instance,
            'filename': self.pts_files[index]
        }

    def collate_fn(self, inputs):
        return [i for i in inputs]

    @staticmethod
    def get_data_loader(self, shuffle=False, batch_size=4):
        return torch.utils.data.DataLoader(self,
                                           batch_size=batch_size,
                                           num_workers=batch_size,
                                           shuffle=shuffle,
                                           collate_fn=self.collate_fn)

    def unified_preload(self, path : str):
        if path.endswith('.bin'):
            return np.fromfile(path, dtype=np.float32).reshape(-1,4)
        elif path.endswith('.npy'):
            return np.load(path, allow_pickle=True)
        elif path.endswith('.label'):
            return NotImplementedError("SemanticKitti format")
        else:
            raise NotImplementedError("Different formats")

    def get_feature(self, idx, name):
        path_to_frame = self.pts_files[idx]
        path_to_feature = glob.glob(os.path.dirname(path_to_frame) + '/../' + name + f'/{idx:06d}*')[0]

        return self.unified_preload(path_to_feature)

    def store_feature(self, feature, idx, name):
        path_to_frame = self.pts_files[idx]
        path_to_feature = os.path.dirname(path_to_frame) + '/../' + name + f'/{idx:06d}.npy'

        os.makedirs(os.path.dirname(path_to_feature), exist_ok=True)

        np.save(path_to_feature, feature)


if __name__ == '__main__':
    dataset = Sequence_Loader(dataset_name='semantic_kitti', sequence=4)
    dataloader = Sequence_Loader.get_data_loader(dataset, batch_size=4)

    # unify the label, instance mapping, mainly for visualization
    for batch in dataloader:
        print(len(batch))
        for folder_type in ['instance', 'label_mapped']:
            seq_root = os.path.dirname(batch[0]['filename'].replace('velodyne', ''))
            os.makedirs(seq_root + '/' + folder_type, exist_ok=True)

            for batch_idx in range(len(batch)):
                np.save(batch[batch_idx]['filename'].replace('velodyne', folder_type).replace('.bin', '.npy'), batch[batch_idx][folder_type])


    # match instnaces without ground for now?

