import numpy as np


"""
def rigid_transformation(pc, mat):
    pc = np.concatenate((pc, np.ones((pc.shape[0], 1))), 1)
    pc = mat @ pc.T
    return pc.T
"""

def rigid_transformation(pc, mat):
    # pc : N x 3
    # mat: 3 x 4
    # In-place transformation
    pc = pc.T
    np.matmul(mat[:, :3], pc, out=pc)
    pc += mat[:, 3:4]
    return pc.T


def calibrate_livox_point_clouds(pc, livox_calib, livox_index=4):
    for i in livox_calib.keys():
        where = pc[:, livox_index] == i
        pc[where, :3] = rigid_transformation(pc[where, :3], livox_calib[i])
    return pc


def project_points_to_camera(pc, cami, came, imsize=None):
    # Transform to camera coordinates system
    pc_proj = np.concatenate((pc[:, :3] * 1000, np.ones((pc.shape[0], 1))), 1)
    pc_proj = (came @ pc_proj.T).T
    depth = pc_proj[:, 2:3] / 1000
    # Perspective projection
    pc_proj[:, 0] = pc_proj[:, 0] / pc_proj[:, 2]
    pc_proj[:, 1] = pc_proj[:, 1] / pc_proj[:, 2]
    # Apply intrinsic transformation
    pc_proj = np.concatenate((pc_proj[:, :2], np.ones((pc_proj.shape[0], 1))), 1)
    pc_proj = (cami @ pc_proj.T).T
    # Restrict to image size
    if imsize is not None:
        where = (
            (pc_proj[:, 0] >= 0)
            & (pc_proj[:, 0] < imsize[1])
            & (pc_proj[:, 1] >= 0)
            & (pc_proj[:, 1] < imsize[0])
            & (depth[:, 0] > 0)
        )
        return np.concatenate((pc_proj[:, :2], depth, pc[:, 3:]), 1)[where]
    else:
        return np.concatenate((pc_proj[:, :2], depth, pc[:, 3:]), 1)
