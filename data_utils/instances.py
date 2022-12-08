import numpy as np
import matplotlib.pyplot as plt

import visualizer

from timespace.trajectory import Trajectory




class Instance3D():
    '''
    Object represented by dictionary of points [values] in frame [keys] and operations on itself
    '''

    def __init__(self, idx=1, class_name='object'):
        # todo remove sequence Len
        super().__init__()
        self.sequence_len = 300
        self.data = {idx: {} for idx in range(self.sequence_len)}
        self.box = None


    def update(self, frame_id: int, stack=False, **kwargs):
        '''
        Use it to delete as if you assign None to the feature
        :param frame_id:
        :param kwargs:
        :return:
        '''
        for key, value in kwargs.items():
            # if stack ...
            self.data[frame_id].update({key : value})

    def full_update(self, **kwargs):
        '''
        Done per frame as it is passed.
        :param kwargs:
        :return:
        '''
        for key, value in kwargs.items():
            for frame_id in range(len(value)):
                self.data[frame_id].update({key: value[frame_id]})

    def compute_trajectory(self, from_modality='odometry'):

        pose_list = []

        for k, v in self.data.items():
            if from_modality in v.keys():
                pose_list.append(v[from_modality])


        if from_modality == 'odometry':
            odometry_array = np.stack(pose_list)
        else:
            odometry_array = np.stack(pose_list)[:,:3,-1]

        xs = odometry_array[:,0]
        ys = odometry_array[:,1]
        zs = odometry_array[:,2]

        self.Trajectory = Trajectory(xs, ys, zs)

        if self.box is not None:
            self.Trajectory.assign_body(self.box)
            # poses are under trajectory class



    def load_object_from_npy(self, path):
        data = np.load(path, allow_pickle=True)
        self.data = data.item()

    def save_object_to_npy(self, path):
        np.save(path, self.data)

    def get_feature(self, feature):
        return [v[feature] for k,v in self.data.items() if feature in v.keys()]


    def visualize_pts(self):
        pcl_list = []

        for k, v in self.data.items():
            if 'pts' in v.keys():
                pcl_list.append(v['pts'])

                if k == 33: #tmp just for presentation
                    break

        visualizer.visualize_multiple_pcls(*pcl_list)


    def plot_time(self, frame_id: int):
        pts = self.data[frame_id]['pts']
        plt.plot(pts[:,0], pts[:,1], 'b.')
        plt.show()


if __name__ == "__main__":
    pass

    first = Instance3D()
    # first.update(frame_id=frame, pts=objects_pts[chosen_obj])

    first.load_object_from_npy('/home/patrik/patrik_data/delft_toy/objects/first.npy')



    # fold back to get the intermediate points?


