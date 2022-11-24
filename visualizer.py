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
        v.set(point_size=0.02)

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
            l.append(n * np.ones(points.shape[0]))

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
    def visualize_points3D(*args):
        pass

if __name__ == "__main__":
        frame_id = int(sys.argv[1])

        import numpy as np
        # import pyviz3d.visualizer as viz
        from pat.toy_dataset import Sequence_Loader

        dataset = Sequence_Loader(sequence=4)
        batch = dataset.__getitem__(frame_id)

        pts = batch['pts']
        instance = batch['instance']
        labels = batch['label_mapped']

        # path = './'
        # instance = np.load(path + f'instance/{frame_id:06}.npy')
        # labels = np.load(path + f'label_mapped/{frame_id:06}.npy')
        # labels[labels==255] = -1
        mask = pts[:,2] > -1.

        visualize_points3D(pts[mask], instance[mask])
        visualize_points3D(pts, mask)
        visualize_points3D(pts[mask], pts[mask,3])

        # pyviz works, use it for remote on rci?
        # for frame_id in range(269, 270):
        #     name = 'PointClouds;' + str(frame_id)
        #     labels = np.load(f'randla_predictions/{frame_id:06d}.npy')
        #     points = np.fromfile(f'velodyne/{frame_id:06d}.bin', dtype=np.float32).reshape(-1, 4)
        #
        #     v = visualize_points3D(points)
        #     v.set(lookat=[0,0,0], r=60)
        #     time.sleep(2)
        #     v.capture(f'/home/patrik/tmp/{frame_id:06d}.png')
        #
        #     time.sleep(2)
        #     v.clear()

            # # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points[:,:3])
            # pcd.colors = o3d.utility.Vector3dVector(labels / 20)
            # vis = o3d.visualization.Visualizer()
            # vis.create_window()
            #
            # vis.add_geometry(pcd)
            #
            # control = vis.get_view_control()
            # control.change_field_of_view(60.0)
            # control.set_front([0.2673283514716997, -0.95566213685799162, 0.12347239641086065])
            # control.set_lookat([0.2673283514716997, -0.95566213685799162, 0.12347239641086065])
            # control.set_up([-0.056565744923280217, 0.11235147526369287, 0.99205718711541346])
            # control.set_zoom(0.049367675781250009)
            #
            # vis.run()
            # vis.capture_screen_image(f'/home/patrik/tmp/{frame_id:06d}.png')
            #
            # vis.destroy_window()
            # del control
            # del vis

