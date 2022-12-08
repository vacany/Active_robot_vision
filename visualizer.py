import os.path

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import socket

if socket.gethostname().startswith("Pat"):
    sys.path.append('/home/patrik/.local/lib/python3.8/site-packages')
    import pptk

    def visualize_points3D(points, labels=None):
        if not socket.gethostname().startswith("Pat"):
            return

        if labels is None:
            v = pptk.viewer(points[:,:3])
        else:
            v = pptk.viewer(points[:, :3], labels)
        v.set(point_size=0.004)

        return v

    def visualize_pcd(file, point_size=0.01):
        import open3d
        points = np.asarray(open3d.io.read_point_cloud(file).points)
        v=pptk.viewer(points[:,:3])
        v.set(point_size=point_size)

    def visualize_voxel(voxel, cell_size=(0.2, 0.2, 0.2)):
        x,y,z = np.nonzero(voxel)
        label = voxel[x,y,z]
        pcl = np.stack([x / cell_size[0], y / cell_size[1], z / cell_size[2]]).T
        visualize_points3D(pcl, label)

    def visualize_poses(poses):
        xyz = poses[:,:3,-1]
        fig, axes = plt.subplots(2, 1)
        axes[0].plot(xyz[:,0], xyz[:,1])
        res = np.abs(poses[:-1, :3, -1] - poses[1:, :3, -1]).sum(1)
        axes[1].plot(res)
        plt.show()

    def visualize_multiple_pcls(*args):
        p = []
        l = []

        for n, points in enumerate(args):
            p.append(points[:,:3])
            l.append(n * np.ones((points.shape[0])))

        p = np.concatenate(p)
        l = np.concatenate(l)
        visualize_points3D(p, l)

    def visualize_plane_with_points(points, n_vector, d):

        xx, yy = np.meshgrid(np.linspace(points[:,0].min(), points[:,0].max(), 100),
                             np.linspace(points[:,1].min(), points[:,1].max(), 100))

        z = (- n_vector[0] * xx - n_vector[1] * yy - d) * 1. / n_vector[2]
        x = np.concatenate(xx)
        y = np.concatenate(yy)
        z = np.concatenate(z)

        plane_pts = np.stack((x, y, z, np.zeros(z.shape[0]))).T

        d_dash = - n_vector.T @ points[:,:3].T

        bin_points = np.concatenate((points, (d - d_dash)[:, None]), axis=1)

        vis_pts = np.concatenate((bin_points, plane_pts))

        visualize_points3D(vis_pts, vis_pts[:,3])

else:
    def visualize_points3D(pts, color=None, path=os.path.expanduser("~") + '/data/tmp_vis/visul'):
        np.save(path, pts)
        if color is None:
            np.save(path + '_color.npy', np.zeros(pts.shape[0], dtype=bool))
        else:
            np.save(path + '_color.npy', color)


        pass
