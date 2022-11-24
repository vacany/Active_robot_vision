import numpy as np
import matplotlib.pyplot as plt
import visualizer



from timespace.geometry import distance_from_points
from timespace.egomotion import Ego_Spread
from timespace.motionflow import Hungarian_point_matching
from pat.toy_dataset import Sequence_Loader

class Spread_instances():

    def __init__(self, dataset):
        self.dataset = dataset

    # def match_ # in ego_motion class? #
    # here it makes sense to do spreading? Or everything in the ego motion? :D

if __name__ == "__main__":
    seq = 0
    dataset = Sequence_Loader('argoverse2_train', sequence=seq)

    Ego_seq = Ego_Spread(dataset)

    Ego_seq.precluster_ids()

    # run this, prepare datasets

    # One object, calculate init velocity as well
    unique_id = 3
    object_mask = clusters == unique_id
    object_points = p[ids>0][object_mask]
    time_points = t[ids>0][object_mask]
    visualizer.visualize_points3D(object_points, time_points)

    t_min = int(time_points.min())
    t_max = int(time_points.max())


    current_object_points = object_points[time_points == t_max]
    # Sample next point cloud
    next_pcl = dataset.__getitem__(t_max + 1)['global_pts']
    # Narrow down the area
    dist_mask = distance_from_points(next_pcl, current_object_points, max_radius=5)
    roi_next_pcl = next_pcl[dist_mask]

    area_indices = np.argwhere(dist_mask)[:, 0]
    next_ind_mask = np.zeros(next_pcl.shape[0], dtype=int)

    # construct cost matrix
    # There is some weird info about non-square matrices
    # BE SURE THAT IT WORKS
    # https://stackoverflow.com/questions/69988671/hungarian-algorithm-in-python-for-non-square-cost-matrices
    # TODO what if there is less points?
    next_indices = Hungarian_point_matching(selected_points=current_object_points,
                             to_match_points=roi_next_pcl,
                             plot=True)


    next_indices_to_orig = area_indices[next_indices]
    next_ind_mask[next_indices_to_orig] = unique_id

    # Save
    dataset.store_feature(next_ind_mask,  t_max+1, name='ego_spread/id_mask/')

    # Apply threshold to stop it

    ### FINDING RIGID TRANSFORMATION BY CHAMFER DISTANCE - WILL BE MORE CONSISTENT?
    # from pytorch3d.loss.chamfer import chamfer_distance
    # import torch
    # min_dist = 5
    # min_idx = [0, 0]
    # for i in range(-10, 10):
    #     print(i)
    #     for j in range(-10, 10):
    #         tensor_cluster1 = torch.tensor(current_object_points, dtype=torch.float).unsqueeze(0)
    #         tensor_cluster2 = torch.tensor(roi_next_pcl, dtype=torch.float).unsqueeze(0)
    #
    #         move_cluster1 = tensor_cluster1.clone()
    #         move_cluster1[0, :, :3] += torch.tensor((i / 10, j / 10, 0))
    #
    #         tmp_chamf_dist = chamfer_distance(move_cluster1, tensor_cluster2)[0]
    #
    #         if tmp_chamf_dist < min_dist:
    #             print(tmp_chamf_dist)
    #             min_dist = tmp_chamf_dist
    #             final_cluster = move_cluster1
    #             min_idx[0] = i
    #             min_idx[1] = j
    #
    #             plt.plot(tensor_cluster1[0, :, 0], tensor_cluster1[0, :, 1], 'b.')
    #             plt.plot(tensor_cluster2[0, :, 0], tensor_cluster2[0, :, 1], 'r.')
    #             plt.plot(final_cluster[0, :, 0], final_cluster[0, :, 1], 'y.')
    #             # plt.savefig(f'/home/vacekpa2/tmp/{min_idx}_{tmp_chamf_dist}_chamfer.png')
    #             plt.show()
    #             plt.clf()

    # fitting the shape - matching, without z movement
