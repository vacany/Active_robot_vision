"""
Module implementing some basic geometry manipulations.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from valplat.util.dtype import X, Y, PHI, obj_dtype


def dist_pts_line(pts, a, b):
    """
    Calculate distances of points to a line defined by points a and b
    :param pts: numpy array of points
    :param a: first points of the line
    :param b: second point of the line
    :return: numpy array of distances of the points from the segment
    """
    ba = b - a
    ptsa = pts - a

    # normal vector
    ba_n = np.array([ba[1], -ba[0]])
    # unity normal vector
    ba_nu = ba_n * (1.0 / np.linalg.norm(ba_n))
    # compute perpendicular distance
    dist = np.abs(np.dot(ptsa, ba_nu))

    return dist


def dist_pts_segment(pts, a, b, fast=False):
    """
    Calculate distances of points to segment defined by points a and b
    :param pts: numpy array of points
    :param a: first points of the segment
    :param b: second point of the segment
    :param fast: If set to True, the distance calculates only perpendicular distance to segment. If the perpendicular
                 line from the point does not intersect with segment, return_if_not_calculated is returned. (1e6)
    :return: numpy array of distances of the points from the segment
    """
    return_if_not_calculated = 1e6

    ba = b - a
    ab = a - b

    ptsa = pts - a
    if np.dot(ba, ba) < 1e-8:
        return np.linalg.norm(ptsa, axis=1)
    ptsb = pts - b
    ptsa_ba = np.dot(ptsa, ba)
    ptsb_ab = np.dot(ptsb, ab)

    out_proj_ba = ptsa_ba < 0.0
    out_proj_ab = ptsb_ab < 0.0

    # normal vector
    ba_n = np.array([ba[1], -ba[0]])
    # unity normal vector
    ba_nu = ba_n * (1.0 / np.linalg.norm(ba_n))
    # compute perpendicular distance
    dist = np.abs(np.dot(ptsa, ba_nu))

    if fast:
        dist[np.logical_or(out_proj_ab, out_proj_ba)] = return_if_not_calculated
    else:
        dist[out_proj_ba] = np.linalg.norm(ptsa[out_proj_ba], axis=1)
        dist[out_proj_ab] = np.linalg.norm(ptsb[out_proj_ab], axis=1)

    return dist


def segment_dist(x0, x1):
    """
    Calculate distance of the segment from the origin. Here, a special case
    of dist_pts_segment is implemented in a slightly optimized form.
    """
    # normal vector
    ba_n = np.array([x1[1] - x0[1], x0[0] - x1[0]])
    dist = intersect(x0, x1, ba_n)
    if dist == np.inf:
        dist = min(np.linalg.norm(x0), np.linalg.norm(x1))
    else:
        # normalize the result
        dist /= np.linalg.norm(ba_n)
    return dist


def dist_pts_polyline(pts, poly, fast=False):
    """
    Calculates list of distances of points from given polyline
    :param pts: list of points to calcualate the distance to
    :param poly: polyline as list of points
    :param fast: parameter handed to dist_pts_segment(), see there for explanation
    :return: numpy array of distances of the points from the segment
    """
    dst = np.zeros(len(pts)) + 1e6
    for i in range(len(poly) - 1):
        dst = np.minimum(dst, dist_pts_segment(pts, poly[i], poly[i + 1], fast=fast))

    return dst


def line_intersection(pt_a1, pt_a2, pt_b1, pt_b2):
    """Calculate intersection of two lines defined by their endpoints"""
    xa1, ya1 = pt_a1
    xa2, ya2 = pt_a2
    xb1, yb1 = pt_b1
    xb2, yb2 = pt_b2

    x = (
        xa1 * (xb2 * (-ya2 + yb1) + xb1 * (ya2 - yb2))
        + xa2 * (xb2 * (ya1 - yb1) + xb1 * (-ya1 + yb2))
    ) / (-((xb1 - xb2) * (ya1 - ya2)) + (xa1 - xa2) * (yb1 - yb2))

    y = (
        xb2 * (ya1 - ya2) * yb1
        + xa1 * ya2 * yb1
        - xb1 * ya1 * yb2
        - xa1 * ya2 * yb2
        + xb1 * ya2 * yb2
        + xa2 * ya1 * (-yb1 + yb2)
    ) / (-((xb1 - xb2) * (ya1 - ya2)) + (xa1 - xa2) * (yb1 - yb2))

    return np.array([x, y])


def intersect(x0, x1, trig):
    """
    Return radial coordinate of the intersection.
    If the two segments are not interceding, inf is returned.
    :param x0: first point of the segment
    :param x1: second point of the segment
    :param trig: (cos(phi), sin(phi)) where phi is the angle of the ray
    :return: distance from the origin to the intersection of the line segment and the ray
    """
    c, s = trig
    denom = (x1[0] - x0[0]) * s - (x1[1] - x0[1]) * c
    if abs(denom) < 1e-3:  # x0, x1, (0, 0) might be (nearly) collinear
        distances = [np.linalg.norm(x0), np.linalg.norm(x1)]
        min_ind = np.argmin(distances)
        diff = np.linalg.norm(x1 - x0)
        if distances[min_ind] + diff == distances[(min_ind + 1) % 2] and not (
            np.sign(s) != np.sign(x0[1]) or np.sign(c) != np.sign(x0[0])
        ):
            # x0, x1, (0, 0) are (nearly) collinear. Second part of above if condition
            # tests a special case where the ray and the segment are collinear but shooting away
            ret = distances[min_ind]
        else:  # special case where the ray is parallel to line x0-x1
            # (or perpendicular, but other direction - away from segment)
            ret = np.inf
    else:
        t = (x0[1] * c - x0[0] * s) / denom
        ret = np.inf
        if -1e-10 <= t <= 1 + 1e-10:
            t = (
                x0[0] * c
                + x0[1] * s
                + (x1[0] - x0[0]) * c * t
                + (x1[1] - x0[1]) * s * t
            )
            ret = t if t > -1e-10 else np.inf
    return ret


def get_inner_points(pts, corners):
    """Return mask of the points inside rectangle."""
    eps = 1e-3
    v, w = corners[1] - corners[0], corners[2] - corners[1]
    v_norm_sq, w_norm_sq = max(v[0] ** 2 + v[1] ** 2, eps), max(
        w[0] ** 2 + w[1] ** 2, eps
    )
    pts = pts - np.add(corners[0], corners[2]) / 2
    if v_norm_sq == eps and w_norm_sq == eps:
        # both directions are ill conditioned -> do not rotate
        v, w = (1, 0), (0, 1)
    elif v_norm_sq == eps:
        # v is ill conditioned -> use w
        v, v_norm_sq = np.divide((-w[1], w[0]), np.sqrt(w_norm_sq)), 1.0
    elif w_norm_sq == eps:
        # w is ill conditioned -> use v
        w, w_norm_sq = np.divide((-v[1], v[0]), np.sqrt(v_norm_sq)), 1.0
    x, y = np.dot(pts, v), np.dot(pts, w)
    v_norm_sq, w_norm_sq = v_norm_sq * 0.5, w_norm_sq * 0.5
    return np.logical_and.reduce(
        (x > -v_norm_sq, x < v_norm_sq, y > -w_norm_sq, y < w_norm_sq)
    )


def get_corners(x=None, phi=None, bb=None, obj=None):
    """Convert offset representation of the box to XY corners.
    Either obj or (x, phi, bb) has to be specified
    The corners are ordered counter-clockwise starting from rear right.
        3---------2\
        |         | 〉
        0---------1/
    """
    if obj is not None:
        x, phi, bb = obj["x"][[0, 1]], obj["x"][2], obj["bb"]
    v = np.array([np.cos(phi), np.sin(phi)])
    w = np.array([-v[1], v[0]])
    return np.array(
        [
            bb[0] * v + bb[1] * w + x,
            bb[2] * v + bb[1] * w + x,
            bb[2] * v + bb[3] * w + x,
            bb[0] * v + bb[3] * w + x,
        ]
    )


def get_obj_from_corners(cor0, cor1, cor2, **kwargs):
    """
    Converts representation of objects from three (arrays of) corners to an (array of) object(s)
    with object dtype used in the platform.
    Each object's orientation is from corner 0 to corner 1.
        2---------X\
        |         | 〉
        0---------1/
    :param cor0: (array of) rear right corner(s)
    :param cor1: (array of) front right corner(s)
    :param cor2: (array of) rear left corner(s)
    the remaining arguments packed in kwargs will be used to set the selected object attributes
    :return: array of objects with centers in the rear right corners
    """
    v, w = np.asarray(cor1) - cor0, np.asarray(cor2) - cor0
    ln, wi = np.linalg.norm(v, axis=-1), np.linalg.norm(w, axis=-1)
    phi = np.arctan2(*v.T[::-1])
    # test for single object
    assert (
        len(v.shape) == 2 or abs(np.cross(v, w) - ln * wi) < 1e-2
    ), "Incorrect corner orientation, see the documentation"
    # test for vector of objects
    assert len(v.shape) == 1 or all(
        abs(np.cross(v, w) - ln * wi) < 1e-2
    ), "Incorrect corner orientation, see the documentation"

    n = len(v) if len(v.shape) == 2 else 1
    objs = np.zeros(n, dtype=obj_dtype)
    objs["x"][:, :2] = cor0
    objs["x"][:, 2] = phi
    objs["bb"][:, 2] = ln
    objs["bb"][:, 3] = wi
    for key, val in kwargs.items():
        objs[key] = val
    return objs if n > 1 else objs[0]


def get_box_offset(ref_pt, phi, obs):
    """
    Calculates box bb offsets from reference points, orientation and a reference point.
    :param ref_pt: Reference point of the object
    :param phi:    Orientation of the object
    :param obs:    Observed point of the object
    :return: bb offsets of the minimal rectangle with respect to the reference points
    """
    # degeneration guard: ill conditioned box does not have well defined
    # ordering of corners. This may cause problems in occlusion model
    bb = 1e-6 * np.array([-1, -1, 1, 1])
    if len(obs) > 1:
        R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        pts = np.dot(obs - ref_pt, R)
        bb += np.concatenate((np.min(pts, axis=0), np.max(pts, axis=0)))
    return bb


def ray_to_segment_intersection(ray_orig, ray_dir, p0, p1, return_distance=False):
    """
    Calculates the intersection point between the given ray and the
    segment defined by the points p0, p1. None is returned if no intersection
    point is found.
    """
    ray_normalized = ray_dir / np.linalg.norm(ray_dir)
    r = intersect(p0 - ray_orig, p1 - ray_orig, ray_normalized)
    if return_distance:
        return r
    return (ray_normalized * r + ray_orig) if r < np.inf else None


def ray_to_poly_intersection(ray_orig, ray_dir, poly):
    """Calculate the distance of point in a given direction from the polyline"""
    return np.min(
        [
            ray_to_segment_intersection(ray_orig, ray_dir, p0, p1, return_distance=True)
            for p0, p1 in zip(poly, poly[1:])
        ]
    )


def world2ego(pts, origin):
    """Convert global coordinates to local coordinates, origin=(x_m, y_m, phi_rad)"""
    c, s = np.cos(origin[2]), np.sin(origin[2])
    return np.dot(pts - origin[:2], [[c, -s], [s, c]])


def ego2world(pts, origin):
    """Convert ego to world coordinates, origin=(x_m, y_m, phi_rad)"""
    c, s = np.cos(origin[2]), -np.sin(origin[2])
    return np.dot(pts, [[c, -s], [s, c]]) + origin[:2]


def transf_pts(pts, transf, inplace=False):
    """
    Converts points using the given 4x4 transformation matrix
    :param pts: array with fields 'x', 'y', 'z' or 'x', 'z' in case 'x' has shape (n, 2)
    :param transf: 4x4 transformation matrix
    :param inplace: boolean argument, if true, change happens in-place
    :return: array of shape (n, 3)
    """
    aux = np.ones(shape=(len(pts), 4), dtype=np.float64)
    if "y" in pts.dtype.names:
        aux[:, 0] = pts["x"]
        aux[:, 1] = pts["y"]
    else:
        aux[:, :2] = pts["x"]
    aux[:, 2] = pts["z"]
    transformed = np.dot(aux, transf.T)[:, :3]
    if inplace:
        if "y" in pts.dtype.names:
            pts["x"] = transformed[:, 0]
            pts["y"] = transformed[:, 1]
        else:
            pts["x"] = transformed[:, :2]
        pts["z"] = transformed[:, 2]
    return transformed


def scan_list_to_gpts(scan_list, odom_list):
    """
    The function reads the scan_list and odom_list and transforms the points to the global coordinates.
    :param scan_list: scan list in local coordinates
    :param odom_list: odometry list in standard valplats format
    :return: points in global coordinate system
    """
    all_gpts = []
    for odom, pts in zip(odom_list["transformation"], scan_list):
        pts_n4 = np.zeros((pts["x"].shape[0], 4), dtype=np.float64)
        pts_n4[:, :2] = pts["x"][:, :]
        pts_n4[:, 2] = pts["z"]
        pts_n4[:, 3] = 1
        gpts = pts_n4 @ odom.T
        all_gpts.append(gpts[:, :3])

    all_gpts = np.concatenate(all_gpts)

    return all_gpts


def quaternion_to_rotmatrix(qw, qx, qy, qz):
    """
    Conversion according to
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    """
    assert np.abs(np.linalg.norm([qw, qx, qy, qz]) - 1) < 1e-6
    mat = np.identity(4)
    mat[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    return mat


def get_sensor_ego_transf(sensor_offsets):
    """
    Returns the transformation from sensor to ego coordinates,
    where sensor origin is given by 3 translations and 3 euler angles
    :param sensor_offsets: 6 sensor offsets (x, y, z, yaw, pitch, roll)[m, rad]
    :return: 4x4 transformation matrix from sensor to ego
    """
    sy, cy = np.sin(-sensor_offsets[3]), np.cos(sensor_offsets[3])
    sp, cp = np.sin(-sensor_offsets[4]), np.cos(sensor_offsets[4])
    sr, cr = np.sin(-sensor_offsets[5]), np.cos(sensor_offsets[5])

    y_mat = np.asarray([[cy, -sy, 0, 0], [sy, cy, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    p_mat = np.asarray([[cp, 0, sp, 0], [0, 1, 0, 0], [-sp, 0, cp, 0], [0, 0, 0, 1]])
    r_mat = np.asarray([[1, 0, 0, 0], [0, cr, -sr, 0], [0, sr, cr, 0], [0, 0, 0, 1]])

    mat = y_mat @ p_mat @ r_mat
    mat[0:3, 3] = sensor_offsets[:3]

    return mat


def get_sensor_offsets(mat):
    """
    Returns sensor origin and euler angles from the 4x4 transformation matrix from sensor to ego coordinates.
    Inverse of get_sensor_ego_transf.
    """
    yaw = np.arctan2(mat[1, 0], mat[0, 0])
    pitch = np.arctan(-mat[2, 0] / np.linalg.norm([mat[2, 1], mat[2, 2]]))
    roll = np.arctan(mat[2, 1] / mat[2, 2])
    return mat[0, 3], mat[1, 3], mat[2, 3], -yaw, -pitch, -roll


def apply_transf2d(pts, matrix):
    """
    Applies the 2D transformation matrix to an array of 2D points .
    :param pts: array of 2D points of shape (n, 2)
    :param matrix: (3, 3)-shaped transformation matrix
    :return: array of 2D points of shape (n, 3) where the third component==1.0
    """
    assert (
        np.shape(pts)[1] == 2
    ), f"Unexpected shape of input array, expected list of 2D points, got {pts.shape}"
    return apply_transf(pts, matrix, same_size=False)


def apply_transf(pts, matrix, same_size=True):
    """
    Applies the transformation matrix to an array of points .
    :param pts: array of points of shape (n, m) or a single point
    :param matrix: (m+1, m+1)-shaped transformation matrix
    :param same_size: if set to false, the dimension of the output is m+1 where the last component==1.0
    :return: array of points or a single point
    """
    n, m = (1, np.shape(pts)[0]) if len(np.shape(pts)) == 1 else np.shape(pts)
    extended = np.ones(shape=(n, m + 1))
    extended[:, :m] = pts
    product = (matrix @ extended.T).T
    product = product / product[:, -1:]
    if same_size:
        product = product[:, :-1]
    return product if len(np.shape(pts)) > 1 else product[0]


def check_intersection(obj1, obj2, tolerance=0.2):
    """
    Check whether the two object are interceeding.
    The algorithm places test points inside obj1 and tests whether some of the
    test points are inside obj2. Sample points are regularly placed along main
    axes of the rectangle, sampling period is determined by 'tolerance' (it can be
    slightly smaller, depends on obj1 size), boundary points are always included
    in the test set. Tolerance should be smaller than the size of obj2
    along both dimensions otherwise, intersection can be missclassified.
    Advice: If obj1 is smaller than obj2, sampling from obj1 will result in
    less test points and the algorithm will be faster.
    """
    test_points = sample_from_object(obj1, tolerance)
    inside_mask = get_inner_points(test_points, get_corners(obj=obj2))
    return np.any(inside_mask)


def sample_from_object(obj, d):
    """
    Get points inside rectangle. The points are regularly placed along main axes,
    sampling step is less or equall to 'd' so that boundary points are always included.
    """
    bb = obj["bb"]
    xrange = np.linspace(bb[0], bb[2], max(int(np.ceil((bb[2] - bb[0]) / d)), 2))
    yrange = np.linspace(bb[1], bb[3], max(int(np.ceil((bb[3] - bb[1]) / d)), 2))
    pts = np.swapaxes(np.meshgrid(xrange, yrange), 0, -1).reshape((-1, 2))
    phi = obj["x"][PHI]
    c, s = np.cos(phi), np.sin(phi)
    return np.dot(pts, [[c, s], [-s, c]]) + obj["x"][[X, Y]]


def rot(mtx, axis, phi):
    """
    Rotates matrix around the given axis by angle phi
    """
    ax = [(axis + 1) % 3, (axis + 2) % 3]
    ax.sort()
    r = np.eye(3, dtype=np.float64)
    idx = ((ax[0], ax[0], ax[1], ax[1]), (ax[0], ax[1], ax[0], ax[1]))
    c, s = np.cos(phi), np.sin(phi)
    if axis == 1:
        s = -s
    r[idx] = [c, -s, s, c]
    return np.dot(r, mtx)


def from_euler(axes, angles):
    """
    Returns a unity matrix rotated by the given Euler angles in radians.
    The rotation order is given by the order in the axes parameter
    (0, 1, and 2 are the possible values in the list)
    """
    mtx = np.eye(3, dtype=np.float64)
    for axis, phi in zip(axes, angles):
        mtx = rot(mtx, axis, phi)
    return mtx


def points_to_line_distance(pts, l0, l1, eps=1e-9):
    """
    Distance from a list of points to a line
    :param p: list of points
    :param l0: line point 1
    :param l1: line point 2
    :param eps: zero divison handling tolerance value
    :return: numpy array of distance values
    """
    line_vect = l1 - l0
    m = np.linalg.norm(line_vect)
    if m > eps:
        proj = np.dot(pts - l0, line_vect[:, None]) / m
        proj_p = l0 + proj * (line_vect / m)
        dist_vects = pts - proj_p
    else:
        dist_vects = pts - l0
    return np.linalg.norm(dist_vects, axis=1)


def simplify_polyline(polyline, epsilon):
    """
    Returns the Ramer Douglas Peucker simplification of the polyline
    :param polyline: Polyline to be simplified
    :param epsilon: Simplification threshold
    :return: Simplified polyline
    """
    if len(polyline) < 3:
        return polyline
    p0, p1 = polyline[0], polyline[-1]
    dist = points_to_line_distance(polyline[1:-1], p0, p1)
    max_dist_index = np.argmax(dist)
    max_dist, index = dist[max_dist_index], max_dist_index + 1
    if max_dist > epsilon:
        return np.vstack(
            (
                simplify_polyline(polyline[: index + 1], epsilon)[:-1],
                simplify_polyline(polyline[index:], epsilon),
            )
        )
    else:
        return np.vstack((p0, p1))


def simplify_closed_polyline(polygon, epsilon):
    """Ramer Douglas Peucker simplification for closed polygons"""
    if len(polygon) <= 3:
        poly_out = polygon
    else:
        nhalf = len(polygon) // 2
        half1 = simplify_polyline(polygon[: nhalf + 1], epsilon)
        half2 = simplify_polyline(polygon[nhalf:], epsilon)
        poly_out = np.concatenate([half1[:-1], half2])
    return poly_out
