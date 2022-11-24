import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


def visualize_connected_points(pts1, pts2, savefig=None):
    plt.plot(pts1[:, 0], pts1[:, 1], 'o')
    plt.plot(pts2[:, 0], pts2[:, 1], 'x')

    for i in range(len(pts1)):
        p = pts1[i]
        r = pts2[i]

        connection = np.array([(p[0], p[1]), (r[0], r[1])])
        plt.plot(connection[:, 0], connection[:, 1], 'y-')

    plt.show()

def visualize_flow3d(pts, velocity, savefig=None):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for idx in range(len(pts)):
        ax.quiver(pts[idx, 0], pts[idx, 1], pts[idx, 2],  # <-- starting point of vector
                  velocity[idx, 0], velocity[idx, 1], velocity[idx, 2],  # <-- directions of vector
                  color='red', alpha=.6, lw=2,
                  )
    plt.show()

def fit_3D_spline():
    # 3D example
    total_rad = 10
    z_factor = 3
    noise = 0.1

    num_true_pts = 200
    s_true = np.linspace(0, total_rad, num_true_pts)
    x_true = np.cos(s_true)
    y_true = np.sin(s_true)
    z_true = s_true/z_factor

    num_sample_pts = 80
    s_sample = np.linspace(0, total_rad, num_sample_pts)
    x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
    y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
    z_sample = s_sample/z_factor + noise * np.random.randn(num_sample_pts)

    tck, u = interpolate.splprep([x_sample,y_sample,z_sample], s=2)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0,1,num_true_pts)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

    fig2 = plt.figure(2)
    ax3d = fig2.add_subplot(111, projection='3d')
    ax3d.plot(x_true, y_true, z_true, 'b')
    ax3d.plot(x_sample, y_sample, z_sample, 'r*')
    ax3d.plot(x_knots, y_knots, z_knots, 'go')
    ax3d.plot(x_fine, y_fine, z_fine, 'g')
    fig2.show()
    plt.show()
