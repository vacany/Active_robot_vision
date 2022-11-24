import numpy as np


def get_ego_bbox():
    ### KITTI EGO Parameters
    l = 3.5
    w = 1.8
    h = 1.73
    x, y, z = 0, 0, -h / 2
    angle = 0
    EGO_BBOX = np.array((x, y, z, l, w, h, angle))

    return EGO_BBOX
