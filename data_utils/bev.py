import dataclasses

import numpy as np
import matplotlib.pyplot as plt


@dataclasses.dataclass
class BEV():

    cell_size : (0.1, 0.1)

    def __init__(self, cell_size=(0.1, 0.1)):
        self.cell_size = cell_size

    def calculate_boundaries(self, points):

        self.x_min = points[:, 0].min()
        self.y_min = points[:, 1].min()

        self.x_max = points[:, 0].max()
        self.y_max = points[:, 1].max()


    def calculate_shape(self):

        self.shape = np.array(
                (np.round((self.x_max - self.x_min) / self.cell_size[0]) + 1,
                 np.round((self.y_max - self.y_min) / self.cell_size[1]) + 1),
                dtype=int)

    def return_point_coordinates(self, points):
        xy_points = np.array(np.round(((points[:, :2] - np.array((self.x_min, self.y_min))) / self.cell_size)), dtype=int)

        return xy_points

    def create_bev_template_from_points(self, *args):
        points = np.concatenate([pts[:,:3] for pts in args])
        self.calculate_boundaries(points)
        self.calculate_shape()

        self.grid = np.zeros(self.shape)

    def generate_bev(self, points, features):
        # bev coordinates
        xy_points = self.return_point_coordinates(points)
        xy_bev = np.zeros(self.shape)
        xy_bev[xy_points[:, 0], xy_points[:, 1]] = features

        return xy_bev

    def generate_image(self, points, features):
        xy_points = self.return_point_coordinates(points)
        xy_image = np.zeros((self.shape[0], self.shape[1], 3))
        xy_image[xy_points[:, 0], xy_points[:, 1], :] = features

        return xy_image

    def transfer_features_to_points(self, points, bev_grid):
        xy_points = self.return_point_coordinates(points)

        return bev_grid[xy_points[:, 0], xy_points[:, 1]]


def compare_points_to_static_scene(pcls, points, cell_size):

    Bev = BEV(cell_size=(cell_size[0], cell_size[1]))
    Bev.create_bev_template_from_points(*[pcls, points])
    cell_z = cell_size[2]

    z_iter = np.round((pcls[:, 2].max() - pcls[:, 2].min()) / cell_z)
    z_min = pcls[:,2].min()
    inside_mask = np.zeros(points.shape[0])

    for z_idx in range(int(z_iter)):
        z_range_mask_pcls = (pcls[:, 2] > (z_min + z_idx * cell_z)) &\
                            (pcls[:, 2] < (z_min + (z_idx + 1) * cell_z))
        z_range_mask_points = (points[:, 2] > (z_min + z_idx * cell_z)) &\
                              (points[:, 2] < (z_min + (z_idx + 1) * cell_z))
        masked_points = pcls[z_range_mask_pcls]

        bev_grid = Bev.generate_bev(masked_points, features=1)

        inside_mask[z_range_mask_points] += Bev.transfer_features_to_points(points[z_range_mask_points], bev_grid)

    return inside_mask




def ego_position(x_range=(0,70), y_range=(-40, 40), cell_size=(0.25, 0.25)):
    ego = (-x_range[0], -y_range[0])
    xy_ego = (ego[0] / cell_size[0], ego[1] / cell_size[1])
    xy_ego = np.array(xy_ego, dtype=np.int)
    return xy_ego

def mask_out_of_range_coors(pcl, x_range=(0, 70), y_range=(-40,40), z_range=(-np.inf, np.inf)):
    '''

    :param pcl: point cloud xyz...
    :param x_range:
    :param y_range:
    :param z_range:
    :return: Mask for each point, if it fits into the range
    '''
    mask = (pcl[:, 0] > x_range[0]) & (pcl[:, 0] < x_range[1]) & \
           (pcl[:, 1] > y_range[0]) & (pcl[:, 1] < y_range[1]) & \
           (pcl[:, 2] > z_range[0]) & (pcl[:, 2] < z_range[1])

    return mask


def calculate_pcl_xy_coordinates(pcl, cell_size=(0.1,0.1)):
    '''

    :param pcl: point cloud xy...
    :param cell_size: size of the bin for point discretization
    :param ego: xy-position of ego in meters
    :return: coordinates for each point in bird eye view
    '''
    xy = np.floor(pcl[:, :2] / cell_size).astype('i4')

    return xy

def shift_xy_coors(xy):
    min_x = xy[:,0].min()
    min_y = xy[:,1].min()

    shift = (min_x, min_y)

    xy -= shift

    return xy, shift

def create_tmp_grid(xy):
    '''

    :param xy: shifted XY coordinates
    :return: Creates a -1 grid of maximum size, so the points xy fits to it
    '''
    # grid_shape = (2 * (np.abs(xy[:, 0]).max() + np.abs(xy_ego[0])) + 1, 2 * np.abs((xy[:, 1]).max() + np.abs(xy_ego[1])) + 1)
    grid_shape = (xy[:,0].max() + 1, xy[:,1].max() + 1)
    tmp_grid = - np.ones(grid_shape)

    return tmp_grid


def calculate_shape(x_range=(0,70), y_range=(-40, 40), cell_size=(0.25, 0.25)):
    '''
    :return: get grid shape
    '''
    grid_shape = np.array([(x_range[1] - x_range[0]) / cell_size[0],
                           (y_range[1] - y_range[0]) / cell_size[1]],
                          dtype=np.int)
    return grid_shape



def construct_bev(pcl, pcl_feature, x_range=(0,70), y_range=(-40, 40), cell_size=(0.25, 0.25)):
    '''
    :param pcl_feature: which point cloud channel to encode.
    :return:
    '''
    pcl[:,0] -= x_range[0]
    pcl[:,1] -= y_range[0]

    range_mask = mask_out_of_range_coors(pcl, x_range=x_range, y_range=y_range)

    pcl = pcl[range_mask]
    pcl_feature = pcl_feature[range_mask]

    xy_pcl = calculate_pcl_xy_coordinates(pcl, cell_size=cell_size)

    sort_mask = pcl_feature.argsort() # for maximum value
    xy_pcl = xy_pcl[sort_mask]
    pcl_feature = pcl_feature[sort_mask]

    grid_shape = calculate_shape(x_range, y_range, cell_size)
    bev = np.zeros(grid_shape)

    bev[xy_pcl[:,0], xy_pcl[:,1]] = pcl_feature

    return bev

def normalize_bev(bev):
    bev = (bev - bev.min()) / (bev.max() - bev.min())
    return bev
