import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.spatial import cKDTree as KDTree


class PCDataset(Dataset):
    def __init__(
        self,
        voxel_size=[0.1, 0.1, 0.1],
    ):
        super().__init__()
        self.voxel_size = np.array(voxel_size)

    def load_pc(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        # Load point cloud without transformation
        pc, filename = self.load_pc(index)

        # Discrete coordinates, point features, labels
        coords = ((pc[:, :3] - pc[:, :3].min(0, keepdims=True)) / self.voxel_size).astype("int")
        _, ind = np.unique(coords, axis=0, return_index=True)

        # Find nearest neighbor to unquantize the point cloud
        _, closest_point = KDTree(pc[ind, :3]).query(pc[:, :3], k=1)

        return coords[ind], pc[ind, 3:], filename, pc[:, :3], closest_point


def collate_pointclouds(list_data):
    # Extract all data
    list_of_data = (list(data) for data in zip(*list_data))
    coords_list, feats_list, filenames_list, pc_list, closest_point_list = list_of_data

    # Concatenate batch dimension
    for ind_in_batch, coords in enumerate(coords_list):
        batch_ind = torch.full((coords.shape[0], 1), ind_in_batch, dtype=torch.int)
        coords_list[ind_in_batch] = np.concatenate((coords, batch_ind), axis=1)

    # Adjust index of closest points
    nb_points = 0
    for ind_in_batch in range(len(closest_point_list)):
        closest_point_list[ind_in_batch] += nb_points
        nb_points += coords_list[ind_in_batch].shape[0]
    closest_point = torch.from_numpy(np.hstack(closest_point_list)).long()

    # Concatenate vertically
    coords = torch.from_numpy(np.vstack(coords_list)).int()
    feats = torch.from_numpy(np.vstack(feats_list)).float()

    # Prepare output variables
    out = {
        "coords": coords,
        "feats": feats,
        "filenames": filenames_list,
        "pc_list": pc_list,
        "closest_point": closest_point,
    }
    return out
