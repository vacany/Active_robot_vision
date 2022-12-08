import numpy as np
import matplotlib.pyplot as plt
import torch

from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.transforms import se3_exp_map
from pytorch3d.ops.points_alignment import iterative_closest_point, corresponding_points_alignment
from sklearn.neighbors import NearestNeighbors
from munkres import Munkres

import visualizer
from timespace.visual_utils import visualize_connected_points

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

def pytorch3d_ICP(pts1, pts2, device=0, verbose=False):
    a = torch.tensor(pts1[:, :3], dtype=torch.float).unsqueeze(0).cuda()
    b = torch.tensor(pts2[:, :3], dtype=torch.float).unsqueeze(0).cuda()

    out = iterative_closest_point(a, b, verbose=verbose)
    R = out.RTs.R[0].detach().cpu()
    T = out.RTs.T[0].detach().cpu()
    transformed_pts = out.Xt[0]

    T_mat = torch.eye(4)
    T_mat[:3,:3] = R
    T_mat[:3,-1] = T

    return T_mat, transformed_pts

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

class Fit_rigid_transoform():

    def __init__(self, rot_vec=(0,0,0), trans=(0,0,0), metric='chamfer'):
        self.init_rot_vec = torch.tensor(rot_vec, dtype=torch.float, requires_grad=True)
        self.init_trans = torch.tensor(trans, dtype=torch.float, requires_grad=True)
        self.metric = metric
        # should be in coordinate of center of the object

    def fit_rigid_transform(self, pts1, pts2, sub_samples=50, lr=0.3, max_iteration=100, plot=False):
        '''

        :param pts1: to be fit on the pts2
        :param pts2: final goal of transforming pts1
        :param max_iteration:
        :param plot:
        :return:
        '''
        rot_vec = torch.tensor(self.init_rot_vec, dtype=torch.float, requires_grad=True)
        trans = torch.tensor(self.init_trans, dtype=torch.float, requires_grad=True)

        # init data
        if type(pts1) == np.ndarray:
            pts1 = torch.tensor(pts1, dtype=torch.float)

        if type(pts2) == np.ndarray:
            pts2 = torch.tensor(pts2, dtype=torch.float)

        if len(pts1.shape) != 3:
            pts1 = pts1.unsqueeze(0)

        if len(pts2.shape) != 3:
            pts2 = pts2.unsqueeze(0)

        features1 = pts1[:,:,3:].clone()
        sample_features1 = pts1[:,:,3:].clone()
        move_pts1 = torch.cat((pts1[:,:,:3], torch.ones((pts1.shape[0], pts1.shape[1], 1))), dim=2)

        sample_pts1 = move_pts1.clone()
        sample_pts2 = pts2.clone()

        for epoch in range(max_iteration):

            log_trans = torch.cat((trans, rot_vec)).unsqueeze(0)
            estimated_transmat = se3_exp_map(log_trans)

            # Sub Sampling
            if move_pts1.shape[1] > sub_samples:
                indices = np.random.choice(move_pts1.shape[1], sub_samples)
                sample_pts1 = move_pts1[:, indices]
                sample_features1 = features1[:, indices]

            if pts2.shape[1] > sub_samples:
                indices2 = np.random.choice(pts2.shape[1], sub_samples)
                sample_pts2 = pts2[:, indices2]

            move_cluster1 = torch.bmm(sample_pts1, estimated_transmat)[:,:,:3]

            moved_with_features1 = torch.cat((move_cluster1, sample_features1), dim=2)

            tmp_chamf_dist = chamfer_distance(moved_with_features1, sample_pts2)[0]

            loss = tmp_chamf_dist
            loss.backward(retain_graph=True)


            print(estimated_transmat)
            print('Loss: ', f'{loss.item():.2f}', 'Grad: ', rot_vec.grad, trans.grad)


            with torch.no_grad():
                rot_vec -= lr * rot_vec.grad
                trans -= lr * trans.grad

                rot_vec.grad.zero_()
                trans.grad.zero_()

            if plot:

                with torch.no_grad():
                    start_vis = pts1.detach()
                    curr_vis = move_cluster1.detach()
                    plt.plot(start_vis[0, :, 0], start_vis[0, :, 1], 'b.')
                    plt.plot(curr_vis[0, :, 0], curr_vis[0, :, 1], 'g.')
                    plt.plot(sample_pts2[0, :, 0], sample_pts2[0, :, 1], 'r.')
                    plt.title(f'Loss: {loss.item():.2f} \t Grad: {rot_vec.grad} \t {trans.grad}')
                    plt.show()

        if plot:
            visualizer.visualize_multiple_pcls(start_vis[0].detach().numpy(), curr_vis[0].detach().numpy(), sample_pts2[0].detach().numpy())

        return rot_vec, trans

if __name__ == "__main__":
    pass
