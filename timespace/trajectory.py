import scipy.interpolate as interpolate
from scipy.spatial.transform.rotation import Rotation
import matplotlib.pyplot as plt
import numpy as np

import visualizer
from timespace.box import get_boxes_from_ego_poses, get_bbox_points

def construct_transform_matrix(rotvec, trans):
    trans_mat = np.eye(4)

    rot_mat = Rotation.from_rotvec(rotvec).as_matrix()
    trans_mat[:3,:3] = rot_mat
    trans_mat[:3,-1] = trans

    return  trans_mat

def approximate_trajectory(xs, ys):

    tck, u = interpolate.splprep([xs, ys])
    x_i, y_i = interpolate.splev(np.linspace(0, 1, 100), tck)

    plt.plot(x_i, y_i)
    plt.scatter(xs, ys, c='r')
    plt.show()


    idx = np.arange(ys.shape[0])
    idx[-5:] = np.flip(idx[-5:][np.argsort(ys[-5:])])

    tck, u = interpolate.splprep([xs[idx], ys[idx]], s=0)
    x_i, y_i = interpolate.splev(np.linspace(0, 1, 100), tck)

    np.gradient(x_i, y_i)
    plt.plot(x_i, y_i)
    plt.scatter(xs, ys, c='r')
    plt.scatter(xs[-5:], ys[-5:], c='g')


class Trajectory():

    def __init__(self, xs, ys, zs=None, ts=None):
        self.x = xs
        self.y = ys
        self.z = np.zeros(self.x.shape) if zs is None else zs
        self.t = list(range(len(self.x))) if ts is None else ts

        self.__compute_derivatives()


    def update_trajectory(self, xs, ys, zs, ts):

        self.x = np.concatenate((self.x, xs), axis=0)
        self.y = np.concatenate((self.y, ys), axis=0)
        self.z = np.concatenate((self.z, zs), axis=0)
        self.t = np.concatenate((self.t, ts), axis=0)

        self.__compute_derivatives()
        self.velocity = self.get_velocity()
        self.speed = self.get_speed()
        self.tangent = self.get_tangent()
        self.curvature = self.get_curvature()

    def __compute_derivatives(self):
        if len(self.x) == 1:
            return

        self.x_t = np.gradient(self.x)
        self.y_t = np.gradient(self.y)
        self.xx_t = np.gradient(self.x_t)
        self.yy_t = np.gradient(self.y_t)

    def get_velocity(self):
        vel = np.array([[self.x_t[i], self.y_t[i]] for i in range(self.x_t.size)])

        return vel

    def get_speed(self):
        speed = np.sqrt(self.x_t * self.x_t + self.y_t * self.y_t)

        return speed

    def get_tangent(self):
        tangent = np.array([1 / self.get_speed()] * 2).transpose() * self.get_velocity()

        return tangent

    def get_curvature(self):
        # ss_t = np.gradient(self.get_speed())

        curvature_val = np.abs(self.xx_t * self.y_t - self.x_t * self.yy_t) / (self.x_t * self.x_t + self.y_t * self.y_t) ** 1.5

        return curvature_val

    def estimate_next_position(self):
        vel = self.get_velocity()

        x = self.x[-1] + vel[-1, 0]
        y = self.y[-1] + vel[-1, 1]
        z = self.z[-1]
        t = self.t[-1] + 1
        new_centroid = np.array((x,y,z,t))

        return new_centroid
        #TODO do ILqR for estimation? Needed for IRL and looks good in paper?

    def assign_body(self, box):
        ''' pass
        '''

        velocity = self.get_velocity()
        yaws = np.arctan2(velocity[:,1], velocity[:,0])

        self.ego_box = box

        rigid_trans = []

        for idx, yaw in enumerate(yaws):
            transformation_matrix = np.eye(4)
            rot = Rotation.from_rotvec(np.array((0, 0, yaw))).as_matrix()
            translation = np.array((self.x[idx], self.y[idx],self.z[idx]))

            transformation_matrix[:3,:3] = rot
            transformation_matrix[:3,-1] = translation

            rigid_trans.append(transformation_matrix)

        self.poses = np.stack(rigid_trans)
        self.boxes = get_boxes_from_ego_poses(self.poses, ego_box=self.ego_box)
        self.boxes_points = get_bbox_points(self.boxes, feature_values=np.arange(len(self.boxes)))



    def plot(self):
        velocity = self.get_velocity()

        for i in range(0, len(self.x)):
            plt.arrow(self.x[i], self.y[i], velocity[i,0], velocity[i,1], head_width=.1)

        plt.plot(self.x, self.y, 'g.', markersize=12)

        if hasattr(self, 'boxes_points'):
            x_boxes = self.boxes_points[:,0]
            y_boxes = self.boxes_points[:,1]
            plt.plot(x_boxes, y_boxes, 'r.', markersize=3)

        plt.axis('equal')
        plt.show()

    def plot3D(self):
        visualizer.visualize_points3D(self.boxes_points)

if __name__ == '__main__':
    coordinates = np.array(
            [[0., 0.], [0.3, 0.], [1.25, -0.1], [2.1, -0.9], [2.85, -2.3], [3.8, -3.95], [5., -5.75], [6.4, -7.8],
             [8.05, -9.9], [9.9, -11.6], [12.05, -12.85], [14.25, -13.7], [16.5, -13.8], [19.25, -13.35], [21.3, -12.2],
             [22.8, -10.5], [23.55, -8.15], [22.95, -6.1], [21.35, -3.95], [19.1, -1.9]])

    x = coordinates[:,0]
    y = coordinates[:,1]

    Jackal = {'x': 0.,
              'y': 0.,
              'z': 0.,
              'l': 1.5,
              'w': 0.8,
              'h': 0.4,
              'yaw': 0.}

    # approximate_trajectory(xs, ys)
    traj = Trajectory(x, y)
    bbox = list(Jackal.values())
    traj.assign_body(bbox[3], bbox[4], bbox[5])

    # visualizer.visualize_points3D(traj.boxes_points, traj.boxes_points[:,3])
    one_box = traj.boxes_points[traj.boxes_points[:,3] == 19]

    visualizer.visualize_points3D(one_box[:,:3], one_box[:,3])

    # from visualizer import visualize_plane_with_points
    # from sklearn.decomposition import PCA
    # pca_model = PCA(3)
    # pca_model.fit(one_box[:,:3])
    # col_idx = np.argmin(pca_model.explained_variance_)
    #
    # n_vector = pca_model.components_[:, col_idx]
    # d_mean = - n_vector.T @ one_box[:,:3].mean(0)
    # d_dash = - n_vector.T @ one_box[:,:3].T
    #
    # visualize_plane_with_points(one_box[:,:3], n_vector=n_vector, d=d_mean)

