import numpy as np
import matplotlib.pyplot as plt
import os
import glob

DATA_DIR = '/home/patrik/data/stanford3d/Annotations'

files = os.listdir(DATA_DIR)

pcl = []
labels = []


for l in files:
    clz = l.split('_')[0]
    labels.append(clz)

clz_list = np.array(labels)
for idx, clz in enumerate(labels):
    clz_list[clz_list == clz] = idx

clz_list = np.array(clz_list, dtype=int)
instances = range(len(clz_list))

for i in range(len(files)):
    file = open(DATA_DIR + '/' + files[i], 'r')
    label = clz_list[i]

    for l in file.readlines():
        pcl.append(np.array(l.split() + [label] + [instances[i]], dtype=float))

pts = np.stack(pcl)

import pptk
v = pptk.viewer(pts[:,:3], pts[:,3:6] / 255)
v.set(point_size=0.003)

v2 = pptk.viewer(pts[:,:3], pts[:,6])
v2.set(point_size=0.003)

v3 = pptk.viewer(pts[:,:3], pts[:,7])
v3.set(point_size=0.003)
