from scipy.interpolate import interp1d
import numpy as np
from scipy.spatial.transform import Rotation

from valplat.util.dtype import X, Y, PHI
from valplat.util.geom import get_corners

# epsilon
EPS_5_SIGNIFICANT = 1e-5
EPS_4_SIGNIFICANT = 1e-4


def compute_portion(
    sum_high_quality, sum_low_quality, logger, error_reporting_suffix=None
):
    """
    Compute portion when signal has high quality
    value. The following ratio is computed: (hi)/(hi + low).

    Input arguments:
        sum_high_quality (float) - Total distance during which
                                          the signal had high quality
                                          value.
        sum_low_quality (float) - Total distance during which the
                                         signal had low quality value
        logger -  Logger to be used to report errors (if any)
        error_reporting_suffix - What suffix (if any) to use when reporting error to
                                    better describe the values that failed

    Return:
        (float) - The resulting ratio (hi)/(hi + low).
    """
    output = 0.0
    denominator = sum_high_quality + sum_low_quality
    if denominator > 0:
        output = sum_high_quality / denominator
    else:
        logger.error(
            "sum_high_quality{0} ({1}) + sum_low_quality{0} ({2}) <= 0".format(
                error_reporting_suffix, sum_high_quality, sum_low_quality
            )
        )
    return output


def rms(array):
    """
    Compute root mean square value of input array.

    Input arguments:
        array (list) - List of values to compute RMS.

    Return:
        (float) - Root mean square value
    """
    if array:
        return np.sqrt(np.mean([x**2 for x in array]))
    else:
        raise ValueError("Non-empty list is expected.")


def _simpson(x):
    """
    Calculates the array y[i + 2] of definite integrals of f.dt
     on intervals from i/n to (i+2)/n for i = [0, 2, 4, ... n - 1], 0 <= i < n,
     where n = 2k+1 is the length of the input array,
     using the Simpson rule.
    The input array must be odd size and greater than 2.
    :param x: the function values measured in timestamps t = i/n, 0 <= i < n
    :return: array y of length n where:
        y[k] is the definite integral of x from k-2 to k for k = [2, 4, ... n - 1],
        y[k] = 0 otherwise.
    """
    n = len(x)
    assert n > 2
    assert n % 2 == 1

    def valid(i):
        return (i > 1) and (i % 2 == 0)

    def rule(x):
        return (x[0] + 4 * x[1] + x[2]) / 3.0

    zero = np.zeros(shape=x[0].shape)
    return np.array(
        [(rule(x[i - 2 : i + 1]) if valid(i) else zero) for i in range(0, n)]
    )


def match_frames(t_frames_from, t_frames_to):
    """
    For two ordered frame time arrays,
    returns the indices of the elements of t_frames_to with values closest
    to values of t_frames_from.
    """
    if len(t_frames_to) > 1:
        dt = np.diff(t_frames_to)
        assert np.min(dt) >= 0, "t_frames_to must be in non-descending order"
        mean = t_frames_to[:-1] + 0.5 * dt
        return np.searchsorted(mean, t_frames_from)
    return np.zeros_like(t_frames_from, dtype=np.uint64)


search_sorted_nearest = match_frames  # convenient alias


def integrate(x, sampletime):
    """
    Calculates the array of definite integrals of x.dt from 0 to T*i
     for each array index i.
    Approximates the measured function with parabolic arc parts for each interval [i, i+2].
    :param x: array of function values
    :param sampletime:
    :return:  array of definite integrals, the same size as x
    """
    n = len(x)
    i0 = 2 * int((n - 1) / 2)  # index of the last even sample
    n0 = i0 + 1  # number of samples from the beginning to the last even sample
    # calculate integrals for triples starting with even samples
    y = np.zeros(shape=x.shape)
    x0 = x * sampletime
    y[0:n0] = _simpson(x0[0:n0])
    # calculate definite integrals for each even i
    for i in range(2, n0, 2):
        y[i] += y[i - 2]

    # calculate odd samples as averages of even samples
    if n > 2:
        y[1 : n - 1] += 0.5 * (y[0 : n - 2] + y[2:n])
    # use trapezoidal rule for the last odd sample
    if 0 < n0 < n:
        y[n0] = y[n0 - 1] + 0.5 * (x0[n0] + x0[n0 - 1])

    return y


def interp(target_x, source_x, source_y, kind="linear", fill_value="extrapolate"):
    """
    Linear 1D interpolation/extrapolation
    Elements with non-increasing source_x are removed.
    """
    mask = np.append([True], np.diff(source_x) > 0)
    source_x, source_y = np.array(source_x)[mask], np.array(source_y)[mask]
    if len(source_y) > 1:
        ret = interp1d(source_x, source_y, kind=kind, fill_value=fill_value)(target_x)
    else:
        ret = np.full(len(target_x), source_y[0])
    return ret


def interp_transformation(t_target, t_source, transf):
    """
    Interpolation of tranformation matrices.
    """
    transf_interp = np.zeros((len(t_target), 4, 4), "f8")
    transf_interp[:, 3, 3] = 1
    transf_interp[:, :3, -1] = np.transpose(
        [interp(t_target, t_source, pos) for pos in transf[:, :3, -1].T]
    )
    euler_source = Rotation.from_matrix(transf[:, :3, :3]).as_euler("zyx").T
    euler_target = [interp(t_target, t_source, angle) for angle in euler_source]
    if len(t_target) > 0:
        transf_interp[:, :3, :3] = Rotation.from_euler(
            "zyx", np.transpose(euler_target)
        ).as_matrix()
    else:
        # workaround for bug in scipy 1.6.0 - 1.6.3
        # failing minimal example:
        #   from scipy.spatial.transform import Rotation
        #   import numpy as np
        #   Rotation.from_euler('zyx', np.zeros(shape=(0, 3), dtype=np.float64))
        transf_interp[:, :3, :3] = np.zeros(shape=(0, 3, 3), dtype=np.float64)

    return transf_interp


def interp_track(t_target, track, dt_max=None, yaw_wrap=2 * np.pi, kind="linear"):
    """
    Interpolation of object track.
    Linear interpolation is used for box position, remaining part of the state
    is interpolated using constant interpolation. If dt_max is set, only the timestamps
    having counterpart in the original array closer than the threshold are provided. This
    is to keep discontinuity when it is actually intentional.
    """
    # empty objects remain empty
    if track.size == 0:
        return track
    track = track[np.argsort(track["t"])]
    t_source = track["t"]
    # constant interpolation
    idx = search_sorted_nearest(t_target, t_source)
    out = track[idx].copy()
    # shift rotational center when necessary
    if yaw_wrap < 2 * np.pi:
        center_track(track)
    # linear interpolation
    out["x"][:, X] = interp(t_target, t_source, track["x"][:, 0], kind=kind)
    out["x"][:, Y] = interp(t_target, t_source, track["x"][:, 1], kind=kind)
    out["x"][:, PHI] = interp(
        t_target, t_source, wrap_angle(track["x"][:, 2], period=yaw_wrap), kind=kind
    )
    out["t"] = t_target
    for i in range(4):
        out["bb"][:, i] = interp(t_target, t_source, track["bb"][:, i], kind=kind)
    if dt_max is not None:
        dt = np.abs(t_target - t_source[idx])
        out = out[dt <= dt_max]
    return out


def center_track(track):
    """
    Sets the track reference position to the box center while fixing the corners
    """
    corners = np.array([get_corners(obj=stamp) for stamp in track])
    lng_half = np.linalg.norm(corners[:, 0, :] - corners[:, 1, :], axis=1) * 0.5
    wdt_half = np.linalg.norm(corners[:, 1, :] - corners[:, 2, :], axis=1) * 0.5
    track["x"][:, [X, Y]] = np.mean(corners, axis=1)
    track["bb"][:, 0] = -lng_half
    track["bb"][:, 1] = -wdt_half
    track["bb"][:, 2] = lng_half
    track["bb"][:, 3] = wdt_half


def wrap_angle(yaw, period=2 * np.pi):
    """Wrap periodical values to avoid discontinuities."""
    half_period = period * 0.5
    yaw_rate = (np.diff(yaw) + half_period) % period - half_period
    return np.concatenate(([yaw[0]], yaw_rate)).cumsum()


def count_longest_sequence(sequence, val):
    """Return length of the longest sequence of the elements equal to the given value"""
    longest_seq, seq_count = 0, 0
    for el in sequence:
        if el == val:
            seq_count += 1
        elif seq_count:  # stop counting elements, update current max if needed
            longest_seq = max(longest_seq, seq_count)
            seq_count = 0
    longest_seq = max(longest_seq, seq_count)  # process the last one
    return longest_seq


class Interval(object):
    def __init__(self, lower=None, upper=None):
        self.__t0 = lower if lower else 0.0
        self.__t1 = upper if upper else np.inf

    def __get_part(self, t0, t, tmax):
        part = np.array([t0, min(t0 + tmax, self.__t1)])
        if part[1] < self.__t1:
            # not the last part
            part = np.array([t0, min(t0 + t, self.__t1)])
        return part

    def set_upper(self, upper):
        self.__t1 = upper

    def parts(self, part_length=None, max_part_length=None):
        """
        :Description:
        Divides the interval given by lower and upper bound into parts.
        The length of all parts except the last one is exactly [part_length].
        The length of the last part lies within (0, [max_part_length]).

        The interval's lower bound is considered as 0, if set to None.
        The interval's upper bound and the part length are considered as
        positive infinity, if set to None.
        :return: An array of size 2, containing the part's lower and upper bound.
        """
        t = part_length if part_length else np.inf
        tmax = max_part_length if max_part_length else 0
        part = self.__get_part(self.__t0, t, tmax)
        while part[0] < part[1]:
            yield part
            part = self.__get_part(part[1], t, tmax)


def interp_struct(target_timestamps, source_struct, struct_time_key="time"):
    """Constant interpolation of a structure"""
    # empty struct remains empty
    if source_struct.size == 0:
        return source_struct
    source_timestamps = source_struct[struct_time_key]
    # constant interpolation
    idx = search_sorted_nearest(target_timestamps, source_timestamps)
    source_struct = source_struct[idx].copy()
    source_struct[struct_time_key] = target_timestamps
    return source_struct


def interp_struct_linear(
    target_timestamps, source_struct, struct_time_key="time", linear_resample_keys=None
):
    """Constant interpolation for all except linear_resample_keys. Expects "flat" structure keys"""
    constant_interp = interp_struct(target_timestamps, source_struct, struct_time_key)
    source_timestamps = source_struct[struct_time_key]
    if linear_resample_keys is not None:
        for key in linear_resample_keys:
            constant_interp[key] = interp(
                target_timestamps, source_timestamps, source_struct[key]
            )
    return constant_interp


def get_continuous_intervals(flag_list):
    """Get list of (start, stop) indices for intervals where the flag is constant True"""
    idx = np.where(flag_list)[0]
    if idx.size:
        break_points = np.where(np.diff(idx) > 1)[0] + 1
        indices = [idx[[0, -1]] for idx in np.split(idx, break_points)]
    else:
        indices = []
    return indices


def remove_unstable(values, stability_th=1):
    """
    Change values so that only values in continuous intervals longer than stability_th are kept.
    The value in an unstable interval is changed to the value from the last preceding stable interval.
    """
    cluster_ids = np.unique(values)
    intervals = np.zeros(0, dtype=[("idx", "u2", (2,)), ("id", "u1")])
    for cid in cluster_ids:
        ret = get_continuous_intervals(values == cid)
        intervals_tmp = np.zeros(len(ret), dtype=intervals.dtype)
        intervals_tmp["idx"], intervals_tmp["id"] = ret, cid
        intervals = np.append(intervals, intervals_tmp)

    intervals = intervals[np.argsort(intervals["idx"][:, 0])]
    previous_id = intervals[0]["id"]
    valid_flag = np.diff(intervals["idx"]) >= stability_th
    out = values.copy()
    for interval, valid in zip(intervals[1:], valid_flag[1:]):
        if valid:
            previous_id = interval["id"]
        else:
            out[interval["idx"][0] : interval["idx"][1] + 1] = previous_id
    return out


def get_continuous_intervals_multiclass(values):
    """Get list of (start, stop) indices for intervals where the value remains constant"""
    cluster_ids = np.unique(values)
    intervals = np.zeros(0, dtype=[("idx", "u2", (2,)), ("value", cluster_ids.dtype)])
    for cid in cluster_ids:
        ret = get_continuous_intervals([el == cid for el in values])
        intervals_tmp = np.zeros(len(ret), dtype=intervals.dtype)
        intervals_tmp["idx"], intervals_tmp["value"] = ret, [cid]
        intervals = np.append(intervals, intervals_tmp)
    return intervals[np.argsort(intervals["idx"][:, 0])]
