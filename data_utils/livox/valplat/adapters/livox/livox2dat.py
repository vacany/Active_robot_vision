from pathlib import Path

import numpy as np
import msgpack

from valplat.adapters.livox.livox_reader import LvxReader
from valplat.adapters.philosys.ph_prj_creator import PhilosysProjectCreator
from valplat.adapters.philosys.extractor_interfaces import (
    AbstractOdoExtractor,
    AbstractScanPointExtractor,
)
from valplat.adapters.philosys.dat_video_reader import ExternalMp4VideoReader
from valplat.util.math import search_sorted_nearest
from valplat.util.path import BSD_ANNOTATION_PROJECT_PATH  # for demonstration purposes
from valplat.util.unit_conversions import NS_2_S
from valplat.util.geom import get_sensor_ego_transf


class LvxWrapper(AbstractScanPointExtractor, AbstractOdoExtractor):
    """Small compatibility layer required by PhilosysProjectCreator"""

    def __init__(
        self, input_lvx_path, lidar_stream_names=None, extrinsics=None, *args, **kwargs
    ):
        """
        lidar_stream_names: serial numbers of livox lidars to be used.
            All streams are used if not specified
        """
        super().__init__(*args, **kwargs)
        self.input_lvx_path = input_lvx_path
        self._raw_stream_names = lidar_stream_names
        self.extrinsics = extrinsics or (0, 0, 0, 0, 0, 0)
        self.cal = get_sensor_ego_transf(self.extrinsics)

    def get_scan_streams(self):
        return ["Scan"]

    def get_scan_list(self, stream_names, t0, t1):
        # merge all scans in the lvx file if the subset to be used is not specified
        stream_names = self._raw_stream_names or LvxReader().get_stream_names(
            self.input_lvx_path
        )
        scan_list = {
            key: LvxReader(stream_name=key).extract(self.input_lvx_path, t0, t1)
            for key in stream_names
        }
        scan_list_merged = []
        for t_master in scan_list[stream_names[0]][0]:
            scan_merged = []
            for t, scan in scan_list.values():
                scan = scan[search_sorted_nearest(t_master, t)]
                scan_xyz = np.zeros_like(scan, PhilosysProjectCreator.sp_dtype)
                scan_xyz["x"], scan_xyz["y"] = scan["x"][:, 0], scan["x"][:, 1]
                scan_xyz["z"], scan_xyz["i"] = scan["z"], scan["i"]

                loc_pts = np.array(
                    [
                        scan_xyz["x"],
                        scan_xyz["y"],
                        scan_xyz["z"],
                        [1 for _ in range(len(scan_xyz["x"]))],
                    ]
                )
                transf_pts = self.cal @ loc_pts
                scan_xyz["x"], scan_xyz["y"], scan_xyz["z"] = (
                    transf_pts[0, :],
                    transf_pts[1, :],
                    transf_pts[2, :],
                )

                scan_merged.append(scan_xyz)
            scan_merged = np.concatenate(scan_merged)
            scan_list_merged.append((t_master, scan_merged))
        return scan_list_merged

    def get_odo_sample(self, t):
        return np.eye(4)  # LVX file does not contain odometry


class LvxBinWrapper(AbstractScanPointExtractor, AbstractOdoExtractor):
    """Reader for bin format provided by RoboAuto"""

    FRAME_TIME_S = 0.1

    def __init__(
        self, input_lvx_path, lidar_stream_names=None, extrinsics=None, *args, **kwargs
    ):
        """
        lidar_stream_names: serial numbers of livox lidars to be used.
            All streams are used if not specified
        """
        super().__init__(*args, **kwargs)
        self.input_lvx_path = input_lvx_path
        self._raw_stream_names = lidar_stream_names
        self.extrinsics = extrinsics or (0, 0, 0, 0, 0, 0)
        self.cal = get_sensor_ego_transf(self.extrinsics)

    def get_scan_streams(self):
        return ["Scan"]

    def get_scan_list(self, stream_names, t0, t1):
        with open(self.input_lvx_path, "rb") as ipt:
            unpacker = msgpack.Unpacker(ipt)
            times = []
            scans = []
            while True:
                try:
                    m = unpacker.unpack()
                    times.append(
                        m[1]["time"].seconds + m[1]["time"].nanoseconds * NS_2_S
                    )
                    scan_xyz = np.zeros(
                        len(m[1]["points"]), dtype=PhilosysProjectCreator.sp_dtype
                    )
                    calibrated_points = np.array(m[1]["points"])
                    if len(calibrated_points) > 0:
                        calibrated_points[:, -1] = 1
                        calibrated_points = (self.cal @ calibrated_points.T).T

                    scan_xyz["x"] = [p[0] for p in calibrated_points]
                    scan_xyz["y"] = [p[1] for p in calibrated_points]
                    scan_xyz["z"] = [p[2] for p in calibrated_points]
                    scan_xyz["i"] = [p[3] for p in m[1]["points"]]
                    scans.append(scan_xyz)

                except msgpack.exceptions.OutOfData:
                    break

        # slicing
        tstep = self.FRAME_TIME_S
        nn = int(np.round(tstep / np.median(np.diff(times))))
        scans = [np.concatenate(scans[n : n + nn]) for n in range(len(times))[::nn]]
        times = [np.mean(times[n : n + nn]) for n in range(len(times))[::nn]]
        times = np.array(times) - times[0]

        return [(t, scan) for t, scan in zip(times, scans)]

    def get_odo_sample(self, t):
        return np.eye(4)  # LVX file does not contain odometry


def lvx_to_dat(
    input_dat_path,
    output_root,
    time_bouds,
    roboauto_bin=False,
    external_video=None,
    extrinsics=None,
):
    if roboauto_bin:  # format received from RoboAuto
        reader = LvxBinWrapper(input_dat_path, extrinsics=extrinsics)
    else:
        reader = LvxWrapper(input_dat_path, extrinsics=extrinsics)
    with ExternalMp4VideoReader(
        [], input_streams=("Video1"), external_video=external_video
    ) as velo_video:
        creator = PhilosysProjectCreator(
            scan_reader=reader,
            odo_reader=reader,
            video_reader=velo_video,
            input_intensity_range=1,
        )
        creator.create(
            Path(input_dat_path).stem,
            output_root,
            structure_path=BSD_ANNOTATION_PROJECT_PATH
            / "BSD_label_project_structure.xml",
            filter_path=BSD_ANNOTATION_PROJECT_PATH / "BSD_label_project_filter.xml",
            input_t0=time_bouds[0],
            input_t1=time_bouds[1],
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from valplat.util.log import initialize_logging

    initialize_logging()
    argp = ArgumentParser()
    argp.add_argument(
        "-i", "--input-lvx-path", required=True, help="Input .lvx file path."
    )
    argp.add_argument("-o", "--output_root", required=True, help="Output tree root")
    argp.add_argument(
        "-t",
        "--time-bounds",
        required=False,
        nargs=2,
        type=float,
        default=[0.0, np.inf],
        help="Timestamp range [s]",
    )
    argp.add_argument("-v", "--video", required=False, help="Video mkv", default=None)
    argp.add_argument(
        "-c",
        "--calibration",
        required=False,
        help="Calib parameters x, y, z, yaw, pitch, roll",
        default=None,
        nargs=6,
        type=float,
    )
    argp.add_argument(
        "--roboauto",
        required=False,
        help="Roboauto format",
        action="store_const",
        const=True,
        default=False,
    )
    args = argp.parse_args()
    lvx_to_dat(
        args.input_lvx_path,
        args.output_root,
        args.time_bounds,
        roboauto_bin=args.roboauto,
        external_video=args.video,
        extrinsics=args.calibration,
    )
