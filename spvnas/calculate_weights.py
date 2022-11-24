import numpy as np
import glob

from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset

dataset = SemKittiDataset()
cfg = dataset.cfg


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

label_paths = sorted(glob.glob('/mnt/beegfs/gpu/temporary/vacekpa2/semantic-kitti/dataset/sequences/*/labels/*.label'))


weights = np.zeros(len(kept_labels))

reverse_label_name_mapping = {}
label_map = np.zeros(260)
cnt = 0

np.set_printoptions(precision=2)

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

for id, label_path in enumerate(label_paths):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half



    labels = label_map[label & 0xFFFF].astype(np.int64)


    u, c = np.unique(labels, return_counts=True)

    for clz, count in zip(u, c):
        if clz == 255:
            continue

        weights[clz] += count

    print(id)

inv = 1 / weights
clz_weights = inv / inv.sum()
print(clz_weights)

np.save('clz_weights.npy', clz_weights)
