import numpy as np
import open3d as o3d
import copy
import glob

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def handpick_points(pcl):

    if type(pcl) is np.ndarray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

        if pcl.shape[1] == 6:
            pcd.colors = o3d.utility.Vector3dVector(pcl[:, 3:6])

        else:
            pcd.colors = o3d.utility.Vector3dVector(np.random.rand(len(pcd.points), 3))

    else:
        pcd = pcl
    # Visualize cloud and edit
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked_indices = vis.get_picked_points()

    picked_points = np.asarray(pcd.points)[picked_indices]

    return picked_points, picked_indices

# use two pencils
data_dir = '/home/patrik/patrik_data/calibration/'
camera_pts_files = sorted(glob.glob(data_dir + '/camera_pts/*.npy'))
lidar_pts_files = sorted(glob.glob(data_dir + '/velodyne/*.npy'))

camera_pts = np.load(camera_pts_files[0])
lidar_pts = np.load(lidar_pts_files[0])


camera_pts = camera_pts[:, [2, 0, 1, 3, 4, 5]]
camera_pts[:,2] = - camera_pts[:,2]
camera_pts[:,1] = - camera_pts[:,1]
# do it over the radius
camera_pts = camera_pts[(camera_pts[:, 0] < 10) & (camera_pts[:, 0] > 0.3)]    # depth is x now


source_points, _ = handpick_points(camera_pts)
target_points, _ = handpick_points(lidar_pts)


# Sample transforms and get the best match
from scipy.spatial.transform.rotation import Rotation
import itertools
from tqdm import tqdm


# random init - for real exp use eye matrix
# angles
pitch = range(0, 1)
roll = range(0, 1)
yaw = range(-90, 90)

# in cm
x_lim = range(-5, 5)
y_lim = range(-5, 5)
z_lim = range(-10, 10)

#TODO smaller steps
#
rotation_combinations = list(itertools.product(*[pitch, roll, yaw]))
translation_combinations = list(itertools.product(*[x_lim, y_lim, z_lim]))

transformation_matrix = np.eye(4)
cost = 1000

reg_p2p = np.eye(4) # init

for rot_vec in tqdm(rotation_combinations):
    rot_vec = np.array(rot_vec) / 180 * np.pi # to rad

    rotation = Rotation.from_rotvec(rot_vec).as_matrix()

    for translation in translation_combinations:
        translation = np.array(translation) / 10 # back to meters

        reg_p2p[:3, :3] = rotation
        reg_p2p[:3, -1] = translation

        pts = np.insert(source_points.copy(), obj=3, values=1, axis=1)

        alligned_pts = pts.copy() @ reg_p2p.T

        tmp_cost = ((alligned_pts[:,:3] - target_points[:,:3]) ** 2).sum()

        if tmp_cost < cost:
            cost = tmp_cost
            best_rot = rotation
            best_trans = translation

transformation_matrix[:3, :3] = best_rot
transformation_matrix[:3, -1] = best_trans

print("Transformation is:")
print(transformation_matrix)
# draw_registration_result(source, target, reg_p2p.transformation)

gl_pts = camera_pts[:,:4]
gl_pts[:,3] = 1
out = gl_pts @ transformation_matrix.T
from visualizer import visualize_multiple_pcls
visualize_multiple_pcls(out, lidar_pts)


