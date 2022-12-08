import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN

import visualizer
from data_utils import bev
from data_utils.instances import Instance3D

from timespace import box
from timespace.geometry import center_position_and_surrounding, min_square_by_pcl, point_distance_from_hull
from timespace.motionflow import Fit_rigid_transoform, pytorch3d_ICP


class Comparing_Cluster_Ids():

    def __init__(self, decision_distance=0.1):

        self.decision_distance = decision_distance
        # self.check_list = [False for _ in self.init_ids]
        self.min_points = 3

    def compare_two_clusters(self, pts1, pts2):
        '''
        Based on convex hull in x-y plane and distance threshold
        :param pts1:
        :param pts2:
        :param decision_distance:
        :return:
        '''

        try:
            distance_to_points = point_distance_from_hull(pts1, pts2, plot=False)
            overlap = np.min(distance_to_points) < self.decision_distance

        except:
            print('could not do the convex hull ---> low nbr of pts or same coordinates (rounded in projection)')
            overlap = False

        return overlap


    def connect_all(self, pts, clusters):
        '''
        Take points and clusters and rewrite the cluster ids to
        unified overlap ids starting from the beginning.
        :param pts:
        :param clusters:
        :return:
        '''
        connection_dict = {}
        all_ids = np.unique(clusters)
        for tested_id in all_ids:
            if tested_id in [0, -1]:
                continue

            for curr_id in all_ids:
                if curr_id in [tested_id, 0, -1]:
                    continue

                # Main Logic
                if np.sum(clusters == curr_id) < self.min_points:
                    continue

                else:
                    same_id_list = self.find_overlaps(pts, clusters,tested_id) # this stores them
                    connection_dict.update({tested_id : same_id_list})

        new_clusters = clusters.copy()

        for key, values in connection_dict.items():
            for v in values:
                if len(values) > 0:
                    new_clusters[new_clusters == v] = key # This will go forth and back

        return new_clusters


    def find_overlaps(self, pts, clusters, interest_id):
        # remap
        connected_list = []

        for test_id in np.unique(clusters):

            if test_id in [interest_id, 0, -1]:
                continue

            single_cluster = pts[clusters == interest_id]
            next_cluster = pts[clusters == test_id]

            if len(next_cluster) < self.min_points: continue  # use min_points in general

            overlap = self.compare_two_clusters(single_cluster, next_cluster)

            if overlap:
                connected_list.append(test_id)

        return connected_list

    def hdbscan_with_features(self, pts_to_cluster, features=None, find_overlaps=True):
        clusterer = HDBSCAN()

        if features is None:
            clusterer.fit(pts_to_cluster)  # Cluster on all features
        else:
            clusterer.fit(pts_to_cluster[:, features])

        clusters = clusterer.labels_ + 1

        if find_overlaps:
            clusters = self.connect_all(pts_to_cluster, clusters)

        return clusters

    def fine_clustering(self, pts):
        ''' To get the nearest points '''
        completion_fit = DBSCAN(eps=self.decision_distance, min_samples=1)
        completion_fit.fit(pts[:,:3])
        next_clusters = completion_fit.labels_ + 1

        return next_clusters


class Dynamic_Instance(Instance3D):

    def __init__(self, **kwargs):
        super().__init__()

        self.Rigid_Fitter = Fit_rigid_transoform()
        self.Connection_clustering = Comparing_Cluster_Ids()

    def drive_through_objects(self, pts, add_z=0):

        boxes = self.Trajectory.boxes

        object_mask = np.zeros(pts.shape[0], dtype=bool)

        for box_id, ego_box in enumerate(boxes.copy()):
            upper_half_points = box.get_point_mask(pts, ego_box, z_add=(0, add_z)) & (
                    pts[:, 2] > ego_box[2] - ego_box[5] / 4)

            object_mask[upper_half_points] = True

        return object_mask

    def _drive_through_pts(self, pts, upper_z=1, below_z=1):
        id = 1
        boxes = self.get_feature('boxes')
        # for instances
        object_mask = np.zeros(pts.shape[0], dtype=bool)
        road_mask = np.zeros(pts.shape[0], dtype=bool)
        marked_points = np.ones(pts.shape[0], dtype=bool)
        id_points = np.zeros(pts.shape[0], dtype=int)

        for box_id, ego_box in enumerate(boxes.copy()):

            upper_half_points = box.get_point_mask(pts, ego_box, z_add=(0, upper_z)) & (
                    pts[:, 2] > ego_box[2] - ego_box[5] / 4)

            # ^ This can be solved with taking only upper window 2D
            if np.any(upper_half_points[marked_points]):
                id_points[upper_half_points] = id
                marked_points[upper_half_points] = False
            else:
                id += 1

            objects_points = np.argwhere(upper_half_points)
            if len(objects_points) > 0:
                object_mask[objects_points[:, 0]] = True

            if not np.any(upper_half_points):
                low_box_points = box.get_point_mask(pts, ego_box, z_add=(below_z, 0))

                road_points = np.argwhere(low_box_points)
                if len(road_points) > 0:
                    road_mask[road_points[:, 0]] = True

        # data = {'object_mask': object_mask,
        # 'road_mask': road_mask,
        # 'id_mask': id_points,

        return object_mask

    def connect_pts(self, frame, direction, pts):
        own_pts = self.data[frame]['pts']
        area_mask = min_square_by_pcl(pts, own_pts, extend_dist=(.5, .5, 0.1), return_mask=True)
        indices = np.argwhere(area_mask)[:, 0]

        shift_own, shift_area = center_position_and_surrounding(own_pts, surrounding_pts=pts[area_mask])
        T_mat, transformed_pts = pytorch3d_ICP(own_pts, pts[area_mask], verbose=True)

        # ground should be eliminated by area mask
        #TODO THIS NEEDS TO BE SETTLED!!! Fine clustering was previously in RGB-D Data
        # area_clusters = self.Connection_clustering.fine_clustering(pts[area_mask])
        area_clusters = self.Connection_clustering.hdbscan_with_features(pts[area_mask], features=[0,1,2])

        tmp_id = area_clusters.max() + 1
        tmp_pts = np.concatenate((transformed_pts.detach().cpu(), pts[area_mask, :3]), axis=0)
        tmp_clu = np.concatenate((tmp_id * np.ones(own_pts.shape[0]), area_clusters), axis=0)
        connection_list = self.Connection_clustering.find_overlaps(tmp_pts, tmp_clu, tmp_id)
        # Todo make overlaps deterministic
        print(area_clusters)
        if len(connection_list) > 0:
            if len(connection_list) > 1:
                print("CONNECTION LIST has length ", len(connection_list))

            next_object_pts = pts[area_mask]
            next_object_pts = next_object_pts[area_clusters == connection_list[0]]
            # motion features should help by direction of connecting the cluster

            # Assign the cluster to the object
            self.data[frame + direction].update(pts=next_object_pts, pose=T_mat)

    def estimate_static(self, cell_df=0.05):
        '''
        eliminate static points by comparing point position over the time, whether they changed or not in spatial
        :param cell_df: Discretization resolution
        :return:
        '''
        pcls = self.get_feature('pts')
        static_mask_list = []

        for ind in range(len(pcls)):
            curr_pts = pcls[ind]
            rest_pts = pcls[:ind] + pcls[ind + 1:]
            print('Static Estimation ', ind, len(rest_pts))
            static_mask = bev.compare_points_to_static_scene(np.concatenate(rest_pts), curr_pts,
                                                             cell_size=(cell_df, cell_df, cell_df))
            static_mask_list.append(static_mask)

        full_pts = np.concatenate(pcls)
        full_dynamic = np.concatenate(static_mask_list) == False

        return static_mask_list
        # visualizer.visualize_points3D(full_pts[full_dynamic])

class Sequence_with_Object():

    def __init__(self, sequence, robot : Dynamic_Instance, pts_source='lidar', exp_name='toy_exp'):
        self.sequence = sequence
        self.robot = robot
        self.pts_source = pts_source
        self.exp_name = exp_name

        self.Clustering = Comparing_Cluster_Ids()

    def intersection_by_object(self, pts):

        movable_pts = self.robot.drive_through_objects(pts)

        return movable_pts


    def cluster_moveable(self, pts, movable_pts):

        new_clusters = np.zeros(movable_pts.shape, dtype='uint16')

        if np.sum(movable_pts) > 3: # todo config
            movable_indices = np.argwhere(movable_pts)[:,0]

            intersect_cluster = self.Clustering.hdbscan_with_features(pts[movable_pts],
                                                                      features=[0, 1, 2],   # todo config
                                                                      find_overlaps=True)
            new_clusters[movable_indices] = intersect_cluster

        return new_clusters

    def get_frame_objects(self, pts, clusters):

        pts_dict = {}
        for curr_id in np.unique(clusters[clusters > 0]):
            object_pts = pts[clusters == curr_id]
            pts_dict.update({curr_id : object_pts})

        return pts_dict

    def annotate_movable(self, frame_id):

        pts = self.sequence.get_feature(frame_id, self.pts_source)
        pose = self.sequence.get_feature(frame_id, 'pose')
        global_pts = self.sequence.pts_to_frame(pts, pose)

        movable_pts = self.intersection_by_object(global_pts)
        new_clusters = self.cluster_moveable(pts, movable_pts)
        object_pts_dict = self.get_frame_objects(pts, new_clusters)

        self.sequence.store_feature(movable_pts, frame_id, f"{self.exp_name}/move_pts")
        self.sequence.store_feature(new_clusters, frame_id, name=f"{self.exp_name}/merged_clusters")

        return object_pts_dict

# Todo change the icp to minimize features as well

if __name__ == "__main__":
    first = Dynamic_Instance()

    first.load_object_from_npy('/home/patrik/patrik_data/delft_toy/objects/first.npy')

    static = first.estimate_static(cell_df=0.05)
    pcls = first.get_feature('pts')

    visualizer.visualize_points3D(np.concatenate(pcls), np.concatenate(static))
