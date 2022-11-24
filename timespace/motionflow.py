import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from sklearn.neighbors import NearestNeighbors
from munkres import Munkres

from visual_utils import visualize_connected_points

def Hungarian_point_matching(selected_points, to_match_points, plot=False):
    '''

    :param selected_points NxD: points to assing into following point cloud
    :param to_match_points MxD: following point cloud
    :return: mask of indices N corresponding to the following point cloud
    '''
    cost_matrix = np.zeros((len(selected_points), len(to_match_points)))

    for i in range(len(selected_points)):
        cost = np.mean((selected_points[i] - to_match_points) ** 2, axis=1)
        cost_matrix[i] = cost

    m = Munkres()
    indices = m.compute(cost_matrix)

    next_indices = [i[1] for i in indices]

    if plot:
        matched_points = to_match_points[next_indices]
        # plot lines connecting the points
        visualize_connected_points(selected_points, matched_points)

    return next_indices


def numpy_chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist



# From Gilles Puy
def sinkhorn(feature1, feature2, pcloud1, pcloud2, epsilon, gamma, max_iter):
    """
    Sinkhorn algorithm - TAKEN FROM FLOT by VALEO.AI
    Parameters
    ----------
    feature1 : torch.Tensor
        Feature for points cloud 1. Used to computed transport cost.
        Size B x N x C.
    feature2 : torch.Tensor
        Feature for points cloud 2. Used to computed transport cost.
        Size B x M x C.
    pcloud1 : torch.Tensor
        Point cloud 1. Size B x N x 3.
    pcloud2 : torch.Tensor
        Point cloud 2. Size B x M x 3.
    epsilon : torch.Tensor
        Entropic regularisation. Scalar.
    gamma : torch.Tensor
        Mass regularisation. Scalar.
    max_iter : int
        Number of unrolled iteration of the Sinkhorn algorithm.
    Returns
    -------
    torch.Tensor
        Transport plan between point cloud 1 and 2. Size B x N x M.
    """

    # Squared l2 distance between points points of both point clouds
    distance_matrix = torch.sum(pcloud1 ** 2, -1, keepdim=True)
    distance_matrix = distance_matrix + torch.sum(
        pcloud2 ** 2, -1, keepdim=True
    ).transpose(1, 2)
    distance_matrix = distance_matrix - 2 * torch.bmm(pcloud1, pcloud2.transpose(1, 2))
    # Force transport to be zero for points further than 10 m apart
    support = (distance_matrix < 10 ** 2).float()   # TODO important hyperparameter?

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / epsilon) * support

    # Early return if no iteration (FLOT_0)
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob1 = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob2 = (
        torch.ones(
            (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
        )
        / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))

    return T

if __name__ == "__main__":
    from pat.toy_dataset import Sequence_Loader
    from pytorch3d.loss.chamfer import chamfer_distance
    import torch

    frame_id = 260
    # test registration
    dataset = Sequence_Loader(dataset_name='synlidar', sequence=4)
    batch = dataset.__getitem__(frame_id)
    batch2 = dataset.__getitem__(frame_id + 1)

    # Preload
    pts = batch['global_pts']
    instance = batch['instance']
    labels = batch['label_mapped']

    pts2 = batch2['global_pts']
    instance2 = batch2['instance']
    labels2 = batch2['label_mapped']

    cluster_id = 19
    cluster1 = pts[instance == cluster_id]
    cluster2 = pts2[instance2 == cluster_id]

    # below lidar
    z_lidar = batch['pose'][2,-1]  # chosen just single batch!
    cluster1 = cluster1[cluster1[:,2] > z_lidar - 1.]
    cluster2 = cluster2[cluster2[:,2] > z_lidar - 1.]

    
    # transformation = refine_cluster_motion(cluster1, cluster2)

    min_dist = 4
    min_idx = [0, 0]
    for i in range(-40,0):
        print(i)
        for j in range(-10,10):
            tensor_cluster1 = torch.tensor(cluster1).unsqueeze(0)
            tensor_cluster2 = torch.tensor(cluster2).unsqueeze(0)

            move_cluster1 = tensor_cluster1.clone()
            move_cluster1[0,:,:3] += torch.tensor((i/10,j/10,0))

            tmp_chamf_dist = chamfer_distance(move_cluster1, tensor_cluster2)[0]

            if tmp_chamf_dist < min_dist:
                print(tmp_chamf_dist)
                min_dist = tmp_chamf_dist
                final_cluster = move_cluster1
                min_idx[0] = i
                min_idx[1] = j

                plt.plot(tensor_cluster1[0, :, 0], tensor_cluster1[0, :, 1], 'b.')
                plt.plot(tensor_cluster2[0,:,0], tensor_cluster2[0,:,1], 'r.')
                plt.plot(final_cluster[0,:,0], final_cluster[0,:,1], 'y.')
                plt.savefig(f'/home/vacekpa2/tmp/{min_idx}_{tmp_chamf_dist}_chamfer.png')
                plt.clf()
