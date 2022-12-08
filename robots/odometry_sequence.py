import os
import numpy as np

from hdbscan import HDBSCAN

from timespace import box
from timespace import geometry
from data_utils.basics import Basic_Dataprocessor
from data_utils.instances import Object3D
from visualizer import visualize_points3D

# class object_bank
# store pts indices


class Ego_Spread():

    def __init__(self, dataset):
        self.dataset = dataset

        self.road_dict = {i : [] for i in range(len(self.dataset))}
        self.object_dict = {i : [] for i in range(len(self.dataset))}
        self.id_dict = {i : [] for i in range(len(self.dataset))}
        self.all_poses = self.dataset.poses.copy()

        # Define lidar poses
        self.ego_boxes = box.get_boxes_from_ego_poses(self.all_poses, ego_box=self.dataset.ego_box)

        # Store data
        self.store_path = os.path.dirname(self.dataset.pts_files[0]) + '/../ego_spread'
        os.makedirs(self.store_path, exist_ok=True)
        for folder in ['object_mask', 'road_mask', 'id_mask']:
            os.makedirs(self.store_path + '/' + folder, exist_ok=True)


    def load_frame(self, idx):
        frame_data = self.dataset.__getitem__(idx)
        global_pts = frame_data['global_pts']

        return global_pts

    def preannotate_frame(self, idx):
        '''
        Annotation by ego poses. Decided about dynamic object when it is intersected in higher part of ego box.
        When this part of ego box is empty, it label the below points as a road.
        :param idx:
        :return:
        '''
        global_pts = self.load_frame(idx)
        p = global_pts.copy()
        object_mask = np.zeros(global_pts.shape[0], dtype=bool)
        road_mask = np.zeros(global_pts.shape[0], dtype=bool)

        # for instances
        marked_points = np.ones(global_pts.shape[0], dtype=bool)
        id_points = np.zeros(global_pts.shape[0], dtype=int)

        id = 1

        for box_id, ego_box in enumerate(self.ego_boxes.copy()):

            if box_id > idx + 500 or box_id < idx - 500: continue


            # todo also use argoverse to segment humans (downsample one of the lidars) for jackal!
            # todo argoverse pose in on the ground - adjust ego bounding box generation!
            # ego_box[2] -= ego_box[5] / 2 # shift it to the middle of the ego - MAYBE NOT VALID FOR OTHER DATASETS!!!
            ego_points = box.get_bbox_points(ego_box[None, :])[:, :4]

            upper_half_points = box.get_point_mask(global_pts, ego_box) & (global_pts[:,2] > ego_box[2] - ego_box[5] / 4)
            # upper_half_points = geometry.points_in_hull(global_pts, ego_points[:,:3]) & (global_pts[:,2] > ego_box[2]) # Faster, but makes mistakes
            # ^ This can be solved with taking only upper window 2D
            if np.any(upper_half_points[marked_points]):
                id_points[upper_half_points] = id
                marked_points[upper_half_points] = False
            else:
                id += 1

            objects_points = np.argwhere(upper_half_points)
            if len(objects_points) > 0:
                object_mask[objects_points[:,0]] = True

            if not np.any(upper_half_points):
                low_box_points = box.get_point_mask(global_pts, ego_box, z_add=(1,0))

                road_points = np.argwhere(low_box_points)
                if len(road_points) > 0:
                    road_mask[road_points[:,0]] = True

            p = np.concatenate((p, ego_points[:,:4]))   # for visualization


        self.object_dict[idx] = object_mask
        self.road_dict[idx] = road_mask
        self.id_dict[idx] = id_points


    def match_ids(self, start, end):
        '''
        Function takes initial frame with its id_mask and connect it by distance metric with the following.
        It assigns close cluster between the frames the same id.
        :param start:
        :param end:
        :return:
        '''
        # todo encapsulate this to the function? Might be usable for later and can be ablated by metrics
        for t in range(start, end):

            pts = self.dataset.__getitem__(t)['global_pts']
            if t == start:
                id_mask = self.dataset.get_feature(t, 'ego_spread/id_mask')
                self.dataset.store_feature(id_mask, idx=t, name='ego_spread/new_id_mask')
            else:
                id_mask = self.dataset.get_feature(t, 'ego_spread/new_id_mask')

            next_pts = self.dataset.__getitem__(t + 1)['global_pts']
            next_id_mask = self.dataset.get_feature(t + 1, 'ego_spread/id_mask')

            new_id_mask = np.zeros(next_id_mask.shape[0], dtype=int)
            # logic, Here, we need to deal with 2 points for example
            for idx in np.unique(id_mask[id_mask > 0]):
                obj_pts = pts[id_mask == idx]

                if len(obj_pts) < 3: continue
                # match ids based on closet id cluster (minimal distance to point)
                candidate_pts = next_pts[next_id_mask > 0]
                # TODO Apply Chamfer distance with multiple features - more precise and scientific
                distance = geometry.point_distance_from_hull(candidate_pts, obj_pts, plot=False)
                closest_points = np.argmin(distance)
                # TODO add maximal distance threshold, other controll points?
                matched_id = next_id_mask[next_id_mask > 0][closest_points]
                # Assign current idx to next frame mask
                new_id_mask[next_id_mask == matched_id] = idx

            self.dataset.store_feature(new_id_mask, idx=t + 1, name='ego_spread/new_id_mask')

    def preload_data(self):
        for i in range(len(self.dataset)):
            self.object_dict[i] = np.load(self.store_path + f'/object_mask/{i:06d}.npy')
            self.road_dict[i] = np.load(self.store_path + f'/road_mask/{i:06d}.npy')
            self.id_dict[i] = np.load(self.store_path + f'/id_mask/{i:06d}.npy')

    def save_preannotation_data(self):
        for i in range(len(self.dataset)):
            np.save(self.store_path + f'/object_mask/{i:06d}.npy', self.object_dict[i])
            np.save(self.store_path + f'/road_mask/{i:06d}.npy', self.road_dict[i])
            np.save(self.store_path + f'/id_mask/{i:06d}.npy', self.id_dict[i])


    def visualize_sequence(self, start=0, end=100):
        object_list = []
        road_list = []
        pts_list = []
        time_list = []

        self.preload_data()

        for i in range(start, end):
            object_list.append(self.object_dict[i])
            road_list.append(self.road_dict[i])
            pts_list.append(self.dataset.__getitem__(i)['global_pts'])
            time_list.append(np.ones(self.object_dict[i].shape[0]) * i)

        object_mask = np.concatenate(object_list)
        road_mask = np.concatenate(road_list)
        pts = np.concatenate(pts_list)
        times = np.concatenate(time_list)

        # visualize_points3D(pts, road_mask * 1 + object_mask * 2)
        # visualize_points3D(pts, times)

    def precluster_ids(self):
        '''
        Preclustering of ego-annotated objects
        :return:
        '''
        p = np.concatenate([self.dataset.__getitem__(i)['global_pts'] for i in range(0, len(self.dataset))])
        t = np.concatenate([i * np.ones(self.dataset.__getitem__(i)['global_pts'].shape[0]) for i in range(0, len(self.dataset))])
        ids = np.concatenate([self.dataset.get_feature(i, 'ego_spread/id_mask/') for i in range(0, len(self.dataset))])

        ego_labeled_points = np.argwhere(ids > 0)[:,0]
        to_cluster = np.concatenate((p[ego_labeled_points],
                                     t[ego_labeled_points, np.newaxis] * self.dataset.framerate), axis=1) # FRAMERATE
        clusterer = HDBSCAN()
        clusterer.fit(to_cluster)  # Cluster all points from ego-annotation
        clusters = clusterer.labels_ + 1

        print('Clustering of Ego-labels \nFound clusters: ', np.unique(clusters))

        all_id_mask = np.zeros(t.shape[0], dtype=int)
        all_id_mask[ego_labeled_points] = clusters

        for time_idx in np.unique(t):
            frame_mask = all_id_mask[t == time_idx]
            self.dataset.store_feature(frame_mask, idx=int(time_idx), name='ego_spread/hdbscan_id_mask/')

    def run_spread(self, end_frame=-1):
        if end_frame == -1:
            end_frame = len(self.dataset)

        for i in range(0, end_frame):
            print(f"Preannotating frame nbr: {i} ----------")
            self.preannotate_frame(i)

        self.save_preannotation_data()
        print("ID matching started")
        # self.match_ids(0, end_frame - 1)
        self.precluster_ids()
        print("ID matching finished")
        print("Spreading started")
        self.spread_ids_to_other_points()
        print("Spreading finished")

    def spread_ids_to_other_points(self):
        # Start from the masks
        for i in range(len(self.dataset)):
            init_mask = self.dataset.get_feature(idx=i, name="ego_spread/hdbscan_id_mask")
            self.dataset.store_feature(init_mask, idx=i, name="ego_spread/spread_id")

        # different way of looping?
        p = np.concatenate([self.dataset.__getitem__(i)['global_pts'] for i in range(0, len(self.dataset))])
        cluster_ids = np.concatenate([self.dataset.get_feature(i, 'ego_spread/spread_id/') for i in range(0, len(self.dataset))])
        t = np.concatenate([i * np.ones(self.dataset.__getitem__(i)['global_pts'].shape[0]) for i in range(0, len(self.dataset))])
        # some check whether
        # clustering will match inbetween points, so it is just first and last


        for unique_id in np.unique(cluster_ids[cluster_ids > 0]):
            # Get points according to id
            object_mask = cluster_ids == unique_id
            object_times = t[cluster_ids == unique_id]
            object_points = p[object_mask]

            # Create object instance class
            object_instance = Object3D(pts=object_points, time_pts=object_times, instance_id=unique_id)

            t_max = object_instance.get_tmax()
            t_min = object_instance.get_tmin()
            # Sample next point cloud
            while t_max < len(self.dataset) - 1:

                next_pcl = self.dataset.__getitem__(t_max + 1)['global_pts']
                cur_points = object_instance.get_point_at_time(time_idx=t_max)

                # TODO Choose the cylinder based on movement of objects
                dist_mask = geometry.distance_from_points(next_pcl, cur_points, max_radius=4) #TODO max_radius as parameter
                dist_mask = (dist_mask) & (next_pcl[:,2] > cur_points[:,2].min() - 0.1) & (next_pcl[:,2] < cur_points[:,2].max() + 0.3)
                # TODO do the selection of valid indices
                print(f"Current Time: {t_max} \t Nbr_pts: {cur_points.shape[0]} \t Nbr_assign_pts: {np.sum(dist_mask)} \t Nbr_object_pts : {object_instance.pts.shape[0]}")

                if cur_points.shape[0] < 2 or np.sum(dist_mask) < cur_points.shape[0]:
                    break

                next_pcl_valid_mask = np.argwhere(dist_mask)[:,0]
                next_indices = object_instance.connect_next(next_pcl[next_pcl_valid_mask])

                # Add chosen points to the object
                new_time_points = (t_max + 1) * np.ones(next_pcl[dist_mask][next_indices].shape[0])
                object_instance.update(next_pcl[dist_mask][next_indices], new_time_points)

                # Save it
                _mask = self.dataset.get_feature(idx=t_max+1, name='ego_spread/spread_id')

                # Creating tmp mask because indexing array with two masks does not work
                tmp_mask = _mask[next_pcl_valid_mask]
                tmp_mask[next_indices] = unique_id
                _mask[next_pcl_valid_mask] = tmp_mask
                # mask reassigned

                self.dataset.store_feature(_mask, idx=t_max+1, name='ego_spread/spread_id')

                t_max += 1


            while t_min > 1:
                # Just copied from previous while loop and adjusted. Should be in function.
                prev_pcl = self.dataset.__getitem__(t_min - 1)['global_pts']
                cur_points = object_instance.get_point_at_time(time_idx=t_min)

                # TODO Choose the cylinder based on movement of objects
                dist_mask = geometry.distance_from_points(prev_pcl, cur_points,
                                                          max_radius=4)  # TODO max_radius as parameter
                dist_mask = (dist_mask) & (prev_pcl[:, 2] > cur_points[:, 2].min() - 0.1) & (
                            prev_pcl[:, 2] < cur_points[:, 2].max() + 0.3)
                # TODO do the selection of valid indices
                print(
                    f"Current Time: {t_min} \t Nbr_pts: {cur_points.shape[0]} \t Nbr_assign_pts: {np.sum(dist_mask)} \t Nbr_object_pts : {object_instance.pts.shape[0]}")

                if cur_points.shape[0] < 2 or np.sum(dist_mask) < cur_points.shape[0]:
                    break

                prev_pcl_valid_mask = np.argwhere(dist_mask)[:, 0]
                next_indices = object_instance.connect_next(prev_pcl[prev_pcl_valid_mask])

                # Add chosen points to the object
                new_time_points = (t_min - 1) * np.ones(prev_pcl[dist_mask][next_indices].shape[0])
                object_instance.update(prev_pcl[dist_mask][next_indices], new_time_points)

                # Save it
                _mask = self.dataset.get_feature(idx=t_min - 1, name='ego_spread/spread_id')

                # Creating tmp mask because indexing array with two masks does not work
                tmp_mask = _mask[prev_pcl_valid_mask]
                tmp_mask[next_indices] = unique_id
                _mask[prev_pcl_valid_mask] = tmp_mask
                # mask reassigned

                self.dataset.store_feature(_mask, idx=t_min - 1, name='ego_spread/spread_id')

                t_min -= 1

                break


if __name__ == "__main__":

    seq = 0
    dataset = Basic_Dataprocessor('argoverse2_train', sequence=seq)

    # Put down the ideas, refactor

    Ego_seq = Ego_Spread(dataset)

    # Ego_seq.precluster_ids()
    # Ego_seq.run_spread()

    # visualization somewhere else
    p = np.concatenate([dataset.__getitem__(i)['global_pts'] for i in range(0, len(dataset))])
    t = np.concatenate([i * np.ones(dataset.__getitem__(i)['global_pts'].shape[0]) for i in range(0, len(dataset))])

    hdbscan_ids = np.concatenate([dataset.get_feature(i, 'ego_spread/hdbscan_id_mask/') for i in range(0, len(dataset))])
    ego_ids = np.concatenate([dataset.get_feature(i, 'ego_spread/object_mask/') for i in range(0, len(dataset))])
    spread_ids = np.concatenate([dataset.get_feature(i, 'ego_spread/spread_id/') for i in range(0, len(dataset))])

    # Matching the points, different configs
    # Improve motion flow and dynamic?

    v4 = visualize_points3D(p, ego_ids)
    v4.set(show_grid=False, bg_color=[1, 1, 1, 1])

    v3 = visualize_points3D(p, spread_ids)
    v3.set(show_grid=False, bg_color=[1, 1, 1, 1])

    v2=visualize_points3D(p[hdbscan_ids>0], ego_ids[hdbscan_ids>0])
    v2.set(show_grid=False, bg_color=[1,1,1,1])

    v=visualize_points3D(p[spread_ids>0], spread_ids[spread_ids>0])
    v.set(show_grid=False, bg_color=[1,1,1,1])
    # eliminate ground - use z coordinate, box adjustments - incremental increase in size until not points in step

    # get pcls in 40 up to 50

    # import box
    # box_pts = box.get_bbox_points(Ego_seq.ego_boxes)
    # from visualizer import visualize_multiple_pcls
    # visualize_multiple_pcls(dataset.__getitem__(0)['global_pts'], box_pts)
    # id_mask = dataset.get_feature(51, name='ego_spread/id_mask')
    #
    # visualize_points3D(dataset.__getitem__(51)['global_pts'], id_mask)
    #
    # Ego_seq = Ego_Spread(dataset)
    # Ego_seq.run_spread()
    # Ego_seq.visualize_sequence(200,210)

    # p=[]
    # ids=[]
    # times=[]
    # for idx in range(len(dataset)):
    #     p.append(dataset.__getitem__(idx)['global_pts'])
    #     ids.append(dataset.get_feature(idx, 'ego_spread/new_id_mask'))
    #     times.append(idx * np.ones(dataset.get_feature(idx, 'ego_spread/new_id_mask').shape[0]))
    #
    # p = np.concatenate((p))
    # p = np.insert(p, 4, values=np.concatenate(times), axis=0)
    # ids = np.concatenate(ids)
    # np.save('p.npy')


    ### 3D Matching
    # dynamic_pts = []
    # for t in range(50, 52):
    #     global_pts = dataset.__getitem__(t)['global_pts']
    #     object_mask = dataset.get_feature(t, 'ego_spread/object_mask/')
    #
    #     obj_pts = np.insert(global_pts[object_mask.astype(bool)], obj=4, values=t, axis=1)
    #
    #     dynamic_pts.append(obj_pts)
    #
    # pts = np.concatenate(dynamic_pts)
    # # visualize_points3D(pts[:,[0,1,4]], pts[:,3])
    #
    # import torch
    # from motionflow.ops import sinkhorn
    # pc1 = torch.tensor(dynamic_pts[0][None,...])
    # pc2 = torch.tensor(dynamic_pts[1][None,...])
    #
    # transport = sinkhorn(pc1[:, :, 3:], pc2[:, :, 3:],
    #          pc1[:, :, :3], pc2[:, :, :3],
    #          epsilon=0.1, gamma=0.1, max_iter=100)
    #
    #
    # row_sum = transport.sum(-1, keepdim=True)
    #
    # # Estimate flow with transport plan
    # ot_flow = (transport @ pc2[0,:,:3]) / (row_sum + 1e-8) - pc1[0,:,:3]


    # Show results on the robot. Run pre-trained models. Projected ETH dataset
    # Start Active learning framework? propagate labels in global map? Chamfer dist?
    # SPVCNN on the robot data - Jackal, run init network
    # setup prios - for possible meeting valounteering 14 days
