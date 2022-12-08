import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def get_max_size(points):
    if len(points) == 2:
        return np.sqrt(((points[1] - points[0]) ** 2).sum())

    else:
        farthest_points = get_farthest_points(points)

    return np.linalg.norm(farthest_points[0] - farthest_points[1], ord=2) # ord == euclidian


def get_farthest_points(points):

    # Find a convex hull in O(N log N)
    hull = ConvexHull(points)

    # Extract the points forming the hull
    hullpoints = points[hull.vertices, :]

    # Naive way of finding the best pair in O(H^2) time if H is number of points on
    # hull
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')

    # Get the farthest apart points
    bestpair = np.unravel_index(hdist.argmax(), hdist.shape)

    farthest_points = ([hullpoints[bestpair[0]], hullpoints[bestpair[1]]])
    return farthest_points


def distance_from_points(pcl, points, max_radius=10.):
    '''
    :param points: xyz special points, which define area(radius) of interest
    :param pcl: whole point cloud to be eliminated
    :param max_radius:
    :return:
    '''

    # speed up
    mask = min_square_by_pcl(pcl, points, extend_dist=max_radius, return_mask=True)
    true_ids = np.argwhere(mask==True)[:,0]

    for idx in true_ids:
        coors = pcl[:,:2] - points[idx][None,:2]
        distance = np.sqrt(np.sum(coors ** 2, axis=1))
        mask[idx] += distance < max_radius

    dist_mask = mask > 0

    return dist_mask



def min_square_by_pcl(points, pcl, extend_dist=(0.,0.,0.), return_mask=False):
    '''
    :param points: points to eliminate
    :param pcl: point cloud which define square area
    :return: rest of points
    '''

    x_min = pcl[:,0].min()
    x_max = pcl[:,0].max()
    y_min = pcl[:,1].min()
    y_max = pcl[:,1].max()
    z_min = pcl[:,2].min()
    z_max = pcl[:,2].max()

    new_points_mask = (points[:,0] >= x_min - extend_dist[0]) & (points[:,0] <= x_max + extend_dist[0]) \
                    & (points[:,1] >= y_min - extend_dist[1]) & (points[:,1] <= y_max + extend_dist[1]) \
                    & (points[:,2] >= z_min - extend_dist[2]) & (points[:,2] <= z_max + extend_dist[2])



    if return_mask:
        return new_points_mask
    else:
        new_points = points[new_points_mask]
        return new_points

def bev_fill_flood(image):
    points_ = np.transpose(np.where(image))
    hull = ConvexHull(points_)
    deln = Delaunay(points_[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1

    return out_img

def get_boundaries_by_points(points):
    x_min = points[:, 0].min()
    y_min = points[:, 1].min()

    x_max = points[:, 0].max()
    y_max = points[:, 1].max()

    return x_min, x_max, y_min, y_max


# TODO Refactor this and points in hull
# def transfer_features_from_bev(points, bev, cell_size=(0.1,0.1)):


def make_images_from_list(pcl_list : list, cell_size=(0.1,0.1)):

    x_min, x_max, y_min, y_max = get_boundaries_by_points(np.concatenate(pcl_list))

    xy_shape = np.array((np.round((x_max - x_min) / cell_size[0]) + 1, np.round((y_max - y_min) / cell_size[1]) + 1),
                        dtype=int)

    # bev coordinates
    bev_list = []
    for num, points in enumerate(pcl_list):
        xy_bev = np.zeros(xy_shape)
        xy_points = np.array(np.round(((points[:, :2] - np.array((x_min, y_min))) / cell_size)), dtype=int)
        xy_bev[xy_points[:, 0], xy_points[:, 1]] = 1

        bev_list.append(xy_bev)

    return bev_list


def make_image_from_points(points, dilation_iter=0, cell_size=(0.1,0.1)):

    x_min, x_max, y_min, y_max = get_boundaries_by_points(points)

    xy_shape = np.array((np.round((x_max - x_min) / cell_size[0]) + 1, np.round((y_max - y_min) / cell_size[1]) + 1),
                        dtype=int)

    xy_bev = np.zeros(xy_shape)

    # bev coordinates
    xy_points = np.array(np.round(((points[:, :2] - np.array((x_min, y_min))) / cell_size)), dtype=int)
    xy_bev[xy_points[:, 0], xy_points[:, 1]] = 1

    # Dilation to close the objects
    kernel = np.ones((5, 5), np.uint8)

    for i in range(dilation_iter):
        xy_bev = cv2.dilate(xy_bev, kernel, iterations=1)
        xy_bev = cv2.erode(xy_bev, kernel, iterations=1)

    return xy_bev
    # from skimage import morphology
    #
    # out = morphology.medial_axis(out_img)

    #
    #
    # plt.imshow(out_img)
    # plt.show()
    # plt.imshow(out)
    # plt.show()


def points_in_hull(points, area_points, dilation_iter=3, cell_size=(0.1, 0.1), plot=False):
    '''
    Approximation to bird eye view and masking - fast but memory heavy
    :param points: Point to decide wheather in path
    :param area_points: Hull points
    :return:
    '''
    points_inside_mask = np.zeros(points.shape[0], dtype=bool)

    x_min, x_max, y_min, y_max = get_boundaries_by_points(area_points)

    xy_shape = np.array((np.round((x_max - x_min) / cell_size[0]) + 1, np.round((y_max - y_min) / cell_size[1]) + 1), dtype=int)

    xy_bev = np.zeros(xy_shape)

    # bev coordinates
    xy_area_points = np.array( np.round(((area_points[:,:2] - np.array((x_min, y_min))) / cell_size)), dtype=int)
    xy_points = np.array(np.round(((points[:, :2] - np.array((x_min, y_min))) / cell_size)), dtype=int)

    # points that fit to bev - can be reduce to contain only area points for memory reduction
    points_mask = (points[:,0] < x_max) & (points[:,0] > x_min) &\
                  (points[:,1] < y_max) & (points[:,1] > y_min)

    xy_bev[xy_area_points[:,0], xy_area_points[:,1]] = 1

    # Dilation to close the objects
    kernel = np.ones((5, 5), np.uint8)

    for i in range(dilation_iter):
        xy_bev = cv2.dilate(xy_bev, kernel, iterations=1)
        xy_bev = cv2.erode(xy_bev, kernel, iterations=1)

    out_img = xy_bev

    inside_mask = out_img[xy_points[points_mask,0], xy_points[points_mask,1]]
    points_inside_mask[points_mask] = np.array(inside_mask, dtype=bool)

    if plot:
        out_img[xy_points[points_mask, 0], xy_points[points_mask, 1]] = 2
        plt.imshow(out_img)
        plt.show()

    return points_inside_mask

def _points_in_hull_old_(points, area_points, cell_size=(0.1, 0.1)):
    '''
    Approximation to bird eye view and masking - fast but memory heavy
    :param points: Point to decide wheather in path
    :param area_points: Hull points
    :return:
    '''

    tmp_points = np.concatenate((points[:,:3], area_points[:,:3]))

    x_min, x_max, y_min, y_max = get_boundaries_by_points(tmp_points)

    xy_shape =  np.array( (np.round((x_max - x_min) / cell_size[0]) + 1, np.round((y_max - y_min) / cell_size[1]) + 1), dtype=int)

    xy_bev = np.zeros(xy_shape)

    # bev coordinates
    xy_area_points = np.array( np.round(((area_points[:,:2] - np.array((x_min, y_min))) / cell_size)), dtype=int)
    xy_points = np.array(np.round(((points[:, :2] - np.array((x_min, y_min))) / cell_size)), dtype=int)

    xy_bev[xy_area_points[:,0], xy_area_points[:,1]] = 1

    out_img = bev_fill_flood(xy_bev)

    inside_mask = out_img[xy_points[:,0], xy_points[:,1]]
    inside_mask = np.array((inside_mask), dtype=bool)

    return inside_mask


def point_distance_from_hull(points, area_points, plot=False):
    distance_list = []

    hull = ConvexHull(area_points[:,:2])
    hull_points = area_points[hull.vertices, :2]

    poly = Polygon(hull_points)  # need to be after hull

    for p in points:

        point = Point(p[0], p[1])

        dist = poly.exterior.distance(point)

        color = '.b'
        if poly.contains(point):
            dist = - dist
            color = '.r'

        distance_list.append(dist)


        if plot:
            plt.plot(p[0], p[1], color)
            pt = p[:2].copy()
            pt[1] = pt[1] + 0.01
            distLabel = "%.2f" % dist
            plt.annotate(distLabel, xy=pt)

    if plot:
        plt.plot(area_points[:, 0], area_points[:, 1], '.y')
        plt.plot(hull_points[:, 0], hull_points[:, 1], '--r', lw=2)
        plt.title(f"Frob metric {np.linalg.norm(distance_list, ord=None):.2f}")
        plt.axis('equal')
        plt.show()

    return distance_list



def cluster_points(points, eps=0.7, min_samples=3):

    model = DBSCAN(eps=eps, min_samples=min_samples)  # sensitive to clustering;  jiny clustering?
    model.fit(points[:, :3])
    clusters = model.labels_

    return clusters

def scaled_dbscan_clustering(points, eps=0.3, min_samples=3, scalling=(1.,1.,1.)):
    tmp_pcl = points.copy()
    tmp_pcl[:,0] *= scalling[0]
    tmp_pcl[:,1] *= scalling[1]
    tmp_pcl[:,2] *= scalling[2]

    model = DBSCAN(eps=eps, min_samples=min_samples)  # sensitive to clustering;  jiny clustering?
    model.fit(tmp_pcl[:, :3])
    clusters = model.labels_

    return clusters


def get_closest_centroid(centroid, centroid_list):
    cents = np.stack(centroid_list)

    dist = np.sqrt( ((centroid[:2] - cents[:,:2]) ** 2).sum(1))

    idx = np.argmin(dist)

    return idx, dist


def get_centroids_from_cluster(points, clusters):
    centroid_list = []

    for c in sorted(np.unique(clusters)):
        # if c == -1: continue    # Beware of backward mapping!
        xyz_cluster = points[clusters==c, :3]
        clusters_centroid = xyz_cluster.mean(0)
        clusters_centroid = np.insert(clusters_centroid, 3, c)
        #TODO max height for comparison
        centroid_list.append(clusters_centroid)

    return centroid_list

def center_position_and_surrounding(center_pts, surrounding_pts):
    pts1 = center_pts.copy()
    pts2 = surrounding_pts.copy()

    center_x = center_pts[:, 0].mean()
    center_y = center_pts[:, 1].mean()

    # Recenter point clouds
    pts1[:, 0] -= center_x
    pts1[:, 1] -= center_y

    pts2[:, 0] -= center_x
    pts2[:, 1] -= center_y

    return pts1, pts2

def transform_pts(inside_pts, transform_mat):
    tmp_pts = inside_pts.copy()
    tmp_pts[:,3] = 1
    shifted_inside_pts = inside_pts.copy()
    shifted_inside_pts[:,:3] = (tmp_pts[:,:4] @ transform_mat.T)[:,:3]

    return shifted_inside_pts
