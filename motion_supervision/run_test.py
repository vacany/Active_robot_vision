import numpy as np
from data_utils.delft_dataset import Delft_Sequence
from data_utils.argoverse2 import Argoverse2_Sequence
from data_utils import machine_paths

from motion_supervision.transfer import Sequence_with_Object, Dynamic_Instance
from data_utils import bev

from visualizer import *

method_cfg = {'hdbscan_feat' : [0,1,2],
              'hdbscan_overlaps' : True,


                }

def find_objects_by_ego(sequence, pts_source, frame=30):


    Ego = Dynamic_Instance()
    ego_poses = sequence.get_ego_poses()
    Ego.full_update(pose=ego_poses)
    Ego.box = sequence.ego_box
    Ego.compute_trajectory(from_modality='pose')


    Scene = Sequence_with_Object(sequence, Ego, pts_source=pts_source)

    objects_pts = Scene.annotate_movable(frame)

    return objects_pts

if __name__ == '__main__':
    Jackal = {'x': 0.,
              'y': 0.,
              'z': 0.,
              'l': 0.5,
              'w': 0.4,
              'h': 0.4,
              'yaw': 0.}

    # match only inside the intersections
    # sequence = Delft_Sequence()

    DATASET_PATH = machine_paths.argoverse2

    sequence = Argoverse2_Sequence(DATASET_PATH)
    for frame in range(39,40):
        objects_pts = find_objects_by_ego(sequence, frame=frame, pts_source='lidar')
        print(frame, len(objects_pts))
    # trying the spreading from init frame

    for object_id, pts in objects_pts.items():
        first = Dynamic_Instance()
        first.update(frame, pts=pts)
        break

    direction = +1
    # next_pcl = sequence.get_global_pts(frame + direction, 'lidar')
    next_pts = sequence.get_feature(frame + direction, 'lidar')
    # keep it in local?
    first.connect_pts(frame, direction=direction, pts=next_pts)

    print(first.data[frame + 1])
    visualize_points3D(first.data[frame+1]['pts'])
    # first.update(frame_id=frame, pts=objects_pts[chosen_obj])

    # first.load_object_from_npy('/home/patrik/patrik_data/delft_toy/objects/first.npy')
    # first.compute_trajectory(from_modality='pose')
    # first.Trajectory.plot()
    # first.visualize_pts()
    #
    #
    # cell_size = (0.05, 0.05, 0.05)
    # curr_pts = first.data[30]['pts']
    # next_pts = first.data[32]['pts']
    #
    # out_mask = bev.compare_points_to_static_scene(curr_pts, next_pts, cell_size)
    #
    # # Poses are not in the global coordinate frame!
    # p = []
    # for k, v in first.data.items():
    #     if 'pts' in v.keys():
    #         pos_med = np.median(v['pts'][:,:3], axis=0)
    #         p.append(v['pts'])
    #         first.data[k].update({'odometry' : pos_med})
    #
    # first.compute_trajectory(from_modality='odometry')
    # first.Trajectory.plot()

    # # sample next pts
    # direction = -1
    # for frame in range(30, 15, direction):
    #     print(frame)
    #     next_pts = Scene.sequence.get_feature(frame + direction, name='camera_pts')
    #     first.connect_pts(frame, direction= direction, pts=next_pts)
    #     np.save('first.npy', first.data)
    #
    # direction = 1
    # for frame in range(30, 33, 1):
    #     print(frame)
    #     next_pts = Scene.sequence.get_feature(frame + direction, name='camera_pts')
    #     first.connect_pts(frame, direction=direction, pts=next_pts)
    #     np.save('first.npy', first.data)
