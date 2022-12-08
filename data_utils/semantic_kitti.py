import os.path

import numpy as np
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

def get_label_mapping():
    reverse_label_name_mapping = {}
    label_map = np.zeros(260)
    cnt = 0

    for label_id in label_name_mapping:
        if label_id > 250:
            if label_name_mapping[label_id].replace('moving-',
                                                    '') in kept_labels:
                label_map[label_id] = reverse_label_name_mapping[
                    label_name_mapping[label_id].replace('moving-', '')]
            else:
                label_map[label_id] = 255
        elif label_id == 0:
            label_map[label_id] = 255
        else:
            if label_name_mapping[label_id] in kept_labels:
                label_map[label_id] = cnt
                reverse_label_name_mapping[
                    label_name_mapping[label_id]] = cnt
                cnt += 1
            else:
                label_map[label_id] = 255

    reverse_label_name_mapping = reverse_label_name_mapping
    num_classes = cnt

    if not os.path.exists(os.path.expanduser("~") + '/data/semantic_kitti/annotations.yaml'):
        with open(os.path.expanduser("~") + '/data/semantic_kitti/annotations.yaml', 'w') as f:
            yaml.dump(reverse_label_name_mapping, f)

        with open(os.path.expanduser("~") + '/data/semantic_kitti/orig_annotations.yaml', 'w') as f:
            yaml.dump(label_name_mapping, f)

        np.save(os.path.expanduser("~") + '/data/semantic_kitti/label_map.npy', label_map)

    return reverse_label_name_mapping, num_classes




def get_ego_bbox():
    ### KITTI EGO Parameters
    l = 3.5
    w = 1.8
    h = 1.73
    x, y, z = 0, 0, -h / 2
    angle = 0
    EGO_BBOX = np.array((x, y, z, l, w, h, angle))

    return EGO_BBOX


class SemanticKitti_Dataset():
    if dataset_name in ['semantic_kitti', 'synlidar']:
        from data_utils.semantic_kitti import get_ego_bbox
        self.framerate = 0.1
        self.ego_box = get_ego_bbox()
        self.pts_files = sorted(glob.glob(PATH_DICT[dataset_name] + f"{sequence:02d}/velodyne/*.bin"))
        self.label_files = sorted(glob.glob(PATH_DICT[dataset_name] + f"{sequence:02d}/labels/*.label"))

        if dataset_name in ['semantic_kitti']:

            self.poses = np.load(PATH_DICT[dataset_name] + f'{sequence:02d}/sequence_poses.npy')
            self.label_map = np.load(PATH_DICT[dataset_name] + '/../../label_map.npy')

        elif dataset_name in ['synlidar']:
            with open(PATH_DICT[dataset_name] + '/../annotations.yaml', 'r') as f:
                config_dict = yaml.load(f, yaml.Loader)['map_2_semantickitti']
                self.label_map = np.array(list(config_dict.values())).astype(np.int32)

            self.poses = np.zeros(len(self.pts_files))

    def __getitem(self, idx):
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

if __name__ == '__main__':
    reverse_mapping, clz_nbr = get_label_mapping()
