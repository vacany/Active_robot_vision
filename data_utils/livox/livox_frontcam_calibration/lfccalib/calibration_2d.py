import cv2
import numpy as np
from scipy.interpolate import griddata


def get_undistortion_map(cam_param, im_shape):
    """
    Undistorsion map for FrontCam OV.
    The map depends on camera parameters and image shape, so it can be saved on disk
    for faster computation.
    """

    # Regular grid (cartesian and cylinder)
    cart_grid = np.indices(im_shape).reshape(2, -1) - np.array(cam_param[0]).reshape(
        2, 1
    )
    radius = np.linalg.norm(cart_grid, axis=0)
    angle = np.arctan2(cart_grid[1], cart_grid[0])

    # Undistorted radius
    den = 1
    for n, val in enumerate(cam_param[1]):
        den += np.sign(val) * ((val * radius) ** (2 * (n + 1)))
    radius_un = radius / den

    # Undistorted grid
    cart_grid_un = np.vstack((radius_un * np.cos(angle), radius_un * np.sin(angle)))

    # Mapping function for undistorsion via opencv
    map_xy = []
    for i in range(2):
        map_xy.append(
            griddata(
                cart_grid_un.T, cart_grid[i].flatten() + cam_param[0][i], cart_grid.T
            ).reshape(im_shape)[None]
        )
    map_xy = np.concatenate(map_xy, 0).astype("float32")

    return map_xy


def undistort_image(image, undistortion_map):
    return cv2.remap(image, undistortion_map[1], undistortion_map[0], cv2.INTER_LINEAR)
