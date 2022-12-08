from datetime import datetime, timedelta, timezone
import os
import logging

logging.basicConfig(level=logging.INFO)
from struct import unpack
from itertools import zip_longest

from progressbar import ProgressBar
import numpy as np

from valplat.util import unit_conversions as uc
from valplat.util.geom import get_sensor_ego_transf, transf_pts
from valplat.util.dtype import pt_dtype
from valplat.util.math import search_sorted_nearest


# definition of Livox LiDAR packet
PACKET_HEADER_DTYPE = [
    ("device_id", "<B"),
    ("version", "<B"),
    ("slot_id", "<B"),
    ("dev_type", "<B"),
    ("reserved", "<B"),
    ("status", "<I"),
    ("timestamp_type", "<B"),
    ("data_type", "<B"),
    ("timestamp", "<Q"),
]  # timestamp can be either nanoseconds or UTC based on timestamp_type
# supported payload types of livox LiDARs (different pointcloud formats and IMU data)
PAYLOAD_DTYPE = [
    [
        ("x", "<i"),
        ("y", "<i"),
        ("z", "<i"),
        ("reflectivity", "<B"),
    ],  # Cartesian Point Cloud, Single Return
    [
        ("r", "<i"),
        ("theta", "<h"),
        ("phi", "<h"),
        ("reflectivity", "<B"),
    ],  # Spherical Point Cloud, Single Return
    [
        ("x", "<i"),
        ("y", "<i"),
        ("z", "<i"),
        ("reflectivity", "<B"),
        ("tag", "<B"),
    ],  # Cartesian Point Cloud, Single Return
    [
        ("r", "<i"),
        ("theta", "<h"),
        ("phi", "<h"),
        ("reflectivity", "<B"),
        ("tag", "<B"),
    ],  # Spherical Point Cloud, Single Return
    [
        ("x1", "<i"),
        ("y1", "<i"),
        ("z1", "<i"),
        ("reflectivity1", "<B"),
        ("tag1", "<B"),
        ("x2", "<i"),
        ("y2", "<i"),
        ("z2", "<i"),
        ("reflectivity2", "<B"),
        ("tag2", "<B"),
    ],  # Cartesian  Point Cloud, Dual Return
    [
        ("theta", "<h"),
        ("phi", "<h"),
        ("r1", "<i"),
        ("reflectivity1", "<B"),
        ("tag1", "<B"),
        ("r2", "<i"),
        ("reflectivity2", "<B"),
        ("tag2", "<B"),
    ],  # Spherical Point Cloud, Dual Return
    [("gyro", "<f", (3,)), ("acc", "<f", (3,))],
]  # IMU Data
PAYLOAD_ELEMENT_COUNT = [
    100,
    100,
    96,
    96,
    48,
    48,
    1,
]  # number of the payload elements based on the type
# enumerations
DATA_TYPE = {**dict.fromkeys(range(5), "Point Cloud"), 6: "IMU"}
DEVICE_TYPE = {0: "Livox Hub", 1: "Mid-40", 2: "Tele-15", 3: "Horizon"}
TIMESTAMP_TYPE = {0: "None", 1: "PTP", 3: "GPS", 4: "PPS"}  # 2 is reserved
UTC_TIMESTAMP = 3  # GPS timestamp is in UTC format
MAX_REFLECTIVITY = 255  # reflectivity is sent as 1byte


class LivoxLidarReaderBase(object):
    """
    Decoder for Livox Horizon and Tele-15 LiDAR data.
    https://github.com/Livox-SDK/Livox-SDK/wiki/Livox-SDK-Communication-Protocol#_0xA_Update_UTC
    Data transfer specific functionality (file, ethernet, ...) shall be implemented in child classes.
    """

    packet_header_size = np.dtype(PACKET_HEADER_DTYPE).itemsize
    payload_size = {
        idx: np.dtype(dtype).itemsize * length
        for idx, (dtype, length) in enumerate(zip(PAYLOAD_DTYPE, PAYLOAD_ELEMENT_COUNT))
    }

    def __init__(self, apply_ego_trans=True):
        self._log = logging.getLogger(self.__class__.__name__)
        self._sensor_ego_transf = None
        self.apply_ego_transf = apply_ego_trans

    def decode_header(self, raw_packet):
        if len(raw_packet) < self.packet_header_size:
            return None, None  # EOF
        raw_header = raw_packet[: self.packet_header_size]
        header = np.frombuffer(raw_header, PACKET_HEADER_DTYPE)[0]
        if header["timestamp_type"] == UTC_TIMESTAMP:
            year, month, day, hour, us = unpack("<BBBBL", header["timestamp"])
            try:
                timestamp = datetime(
                    2000 + year, month, day, hour, tzinfo=timezone.utc
                ) + timedelta(
                    microseconds=us,
                )  # datetime object
                timestamp = (
                    timestamp - datetime(1970, 1, 1, tzinfo=timezone.utc)
                ).total_seconds()  # cast datetime object to an offset in seconds
            except ValueError:
                # self._log.warning(f'Wrong timestamp, skip {year}, {month}, {day}, {hour}, {us}')
                return header, -1
        else:
            timestamp = header["timestamp"] * uc.NS_2_S
        return header, timestamp

    def decode_payload(self, raw_packet, header):
        raw_header = raw_packet[: self.packet_header_size]
        header = np.frombuffer(raw_header, PACKET_HEADER_DTYPE)[0]
        data_type = header["data_type"]
        # --- Gilles: extract "port id" / "slot id"
        slot_id = header["slot_id"]
        # ---
        payload_size = self.payload_size[data_type]
        raw_payload = raw_packet[self.packet_header_size :][:payload_size]
        if len(raw_payload) < payload_size:
            return None  # EOF
        payload = np.frombuffer(raw_payload, dtype=PAYLOAD_DTYPE[data_type])
        if data_type in (0, 2):  # Cartesian Point Cloud
            scan = np.zeros(payload.shape, dtype=pt_dtype)
            scan["x"][:, 0] = payload["x"] * uc.MM_2_M
            scan["x"][:, 1] = payload["y"] * uc.MM_2_M
            scan["z"] = payload["z"] * uc.MM_2_M
            scan["i"] = payload["reflectivity"] / MAX_REFLECTIVITY
            if self.apply_ego_transf:
                transf_pts(scan, self._sensor_ego_transf, inplace=True)
            # --- Gilles: extract "port id" / "slot id"
            scan["layer_id"] = slot_id
            # ---
            payload = scan
        elif data_type == 4:  # Cartesian Point Cloud Dual return
            scan = np.zeros(payload.shape, dtype=pt_dtype)
            scan["x"][:, 0] = payload["x1"] * uc.MM_2_M
            scan["x"][:, 1] = payload["y1"] * uc.MM_2_M
            scan["z"] = payload["z1"] * uc.MM_2_M
            scan["i"] = payload["reflectivity1"] / MAX_REFLECTIVITY
            if self.apply_ego_transf:
                transf_pts(scan, self._sensor_ego_transf, inplace=True)
            # --- Gilles: extract "port id" / "slot id"
            scan["layer_id"] = slot_id
            # ---
            payload = scan
        elif data_type in (1, 3, 5):  # Spherical Point Cloud
            raise NotImplementedError
        else:
            pass  # IMU data, keep the Livox format
        return payload


class LvxReader(LivoxLidarReaderBase):
    """
    Reader for Livox LiDAR data serialized in LVX file.
    Format definition is from
    https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Download/LVX%20Specifications.pdf
    """

    # LVX specific defines
    public_header_size, private_header_size = 24, 5
    device_info_size = 59
    frame_header_size = 24
    max_packet_size = LivoxLidarReaderBase.packet_header_size + max(
        LivoxLidarReaderBase.payload_size.values()
    )
    magic = 0xAC0EA767
    lvx_version = 5  # supported LVX version

    def __init__(self, frame_time_ms=100, stream_name=None, apply_ego_trans=True):
        """
        stream_name: if provided, only the packets corresponding to the lidar with serial number
            equal to stream_name are provided. This parameter is mandatory for files with multiple streams.
        """
        super().__init__(apply_ego_trans=apply_ego_trans)
        self._frame_time_ms = frame_time_ms
        self._stream_name = stream_name

    def _parse_file_header(self, fstream):
        """Parse LVX header, return device count and frame duration"""
        signature, version, magic = unpack(
            "<16s4sI", fstream.read(self.public_header_size)
        )
        assert (
            magic == self.magic
        ), f"magic word does not match, {self.magic} != {magic}"
        frame_duration, device_count = unpack(
            "<IB", fstream.read(self.private_header_size)
        )
        return device_count, frame_duration

    def _read_dev_info(self, fstream):
        dev_info = fstream.read(self.device_info_size)
        (
            lidar_sn_code,
            hub_sn_code,
            dev_id,
            dev_type,
            extrinsic,
            roll,
            pitch,
            yaw,
            x,
            y,
            z,
        ) = unpack("<16s16sBBBffffff", dev_info)
        lidar_sn_code = lidar_sn_code.decode().strip("\x00")
        return (
            lidar_sn_code,
            hub_sn_code,
            dev_id,
            dev_type,
            extrinsic,
            roll,
            pitch,
            yaw,
            x,
            y,
            z,
        )

    def get_stream_names(self, filename, print_info=False):
        """
        Get list of lidar streams in the file. Some additional info like type and
        mounting position per sensor is displayed ff print_info is set to True.
        Mainly for debugging purposes
        """
        stream_names = []
        with open(filename, "rb") as f:
            device_count, _ = self._parse_file_header(f)
            for dev in range(device_count):
                (
                    lidar_sn_code,
                    hub_sn_code,
                    dev_id,
                    dev_type,
                    extrinsic,
                    roll,
                    pitch,
                    yaw,
                    x,
                    y,
                    z,
                ) = self._read_dev_info(f)
                stream_names.append(lidar_sn_code)
                if print_info:
                    mnt_pos = f"{x:>5.1f}, {y:>5.1f}, {z:>5.1f}, {yaw:>6.2f}, {pitch:>6.2f}, {roll:>6.2f} [m, deg]"
                    self._log.info(
                        f"{dev_id:>02}: {DEVICE_TYPE[dev_type]:>8}({lidar_sn_code}), mnt_pos={mnt_pos}, extrinsic={extrinsic}"
                    )
        return stream_names

    def extract(
        self, filename, t0=-np.inf, t1=np.inf, reset_recording_time_offset=True
    ):
        """Extract the selected lidar stream from the file"""
        with open(filename, "rb") as f:
            device_index, frame_duration = self._read_init_section(f)
            timestamps_s, frames = self._read_data_frames(
                f, device_index, t0, t1, reset_recording_time_offset
            )
        # concatenate frames into individual scans
        step = int(self._frame_time_ms / frame_duration)
        if step == 1:
            scans = frames  # nothing, frames already represents individual scans
        elif step == 2:
            pairs = zip_longest(
                frames[::2], frames[1::2], fillvalue=np.zeros(0, dtype=frames[0].dtype)
            )
            scans = [np.concatenate((frame1, frame2)) for frame1, frame2 in pairs]
            timestamps_s = timestamps_s[::2]
        else:
            raise NotImplementedError(
                f"Frame step {step} is not supported. Valid option is 1 and 2."
            )
        return timestamps_s, scans

    def _read_init_section(self, fstream):
        """Reader file headers and device info. Return index of the device of interest"""
        device_index = None  # index of the stream of interest in the device info list
        device_count, frame_duration = self._parse_file_header(fstream)
        assert (device_count == 1) or (
            self._stream_name is not None
        ), f"Livox stream name has to be specified for multi stream file ({device_count} streams present)"
        for dev in range(device_count):
            (
                lidar_sn_code,
                hub_sn_code,
                dev_id,
                dev_type,
                extrinsic,
                roll,
                pitch,
                yaw,
                x,
                y,
                z,
            ) = self._read_dev_info(fstream)
            if device_count == 1 or lidar_sn_code == self._stream_name:
                self._sensor_ego_transf = get_sensor_ego_transf(
                    (x, y, z, -np.deg2rad(yaw), -np.deg2rad(pitch), -np.deg2rad(roll))
                )
                device_index = dev_id
        assert (
            device_index is not None
        ), f'Livox stream "{self._stream_name}" is not present'
        return device_index, frame_duration

    def _read_data_frames(
        self, fstream, device_index, t0, t1, reset_recording_time_offset
    ):
        """Browse the file and get the data provided by the device of interest"""
        timestamps_s, frames, starttime = [], [], None
        bar = ProgressBar(
            max_value=os.path.getsize(fstream.name)
        )  # monitor progress of the file reading
        while 1:
            ret = self._read_data_frame(fstream, device_index)
            if ret is None:
                break  # EOF
            current_offset, frame_time, frame_pts = ret
            if starttime is None:
                starttime = frame_time
            if reset_recording_time_offset:
                frame_time -= starttime
            if timestamps_s and frame_time < timestamps_s[-1]:
                self._log.warning(
                    f"negative timestamp, skipping (..., {timestamps_s[-2]:.3f}, {timestamps_s[-1]:.3f} -> {frame_time:.3f})"
                )
                continue
            if t0 <= frame_time:
                frames.append(frame_pts)
                timestamps_s.append(frame_time)
            if frame_time > t1:
                break  # EOF or end of interval of interest
            bar.update(current_offset)
        bar.finish()  # reading of the file is done
        return timestamps_s, frames

    def _read_data_frame(self, fstream, device_index):
        """
        Return time and scanpoints points in the current data frame.
        Timestamp of the data frame is the timestamp of the first packet in the frame.
        """
        frame_header = fstream.read(self.frame_header_size)
        if len(frame_header) < self.frame_header_size:
            return  # EOF
        current_offset, next_offset, frame_idx = unpack("<QQQ", frame_header)
        current_offset += self.frame_header_size
        frame_pts, frame_time = [], []
        while current_offset < next_offset:
            fstream.seek(current_offset)
            raw_packet = fstream.read(self.max_packet_size)
            header, timestamp = self.decode_header(raw_packet)
            if header is None:
                return  # EOF
            if header["version"] != self.lvx_version:
                # the rest of the data frame is corrupted -> do not use the packets inside
                current_offset = next_offset
                break
            elif (
                DATA_TYPE.get(header["data_type"]) == "Point Cloud"
                and header["device_id"] == device_index
            ):
                payload = self.decode_payload(raw_packet, header)
                if payload is None:
                    return  # EOF
                frame_time.append(timestamp)
                frame_pts.append(payload)
            current_offset += self.packet_header_size + self.payload_size.get(
                header["data_type"], next_offset
            )
        if len(frame_time) == 0:  # to avoid crash in case of corrupted data frame
            frame_time, frame_pts = [0], [[]]
        return (
            fstream.seek(min(next_offset, current_offset)),
            frame_time[0],
            np.concatenate(frame_pts),
        )


class LivoxDatReader(LivoxLidarReaderBase):
    """Reader for Livox LiDAR data serialized in DAT file as raw ethernet stream."""

    def extract(self, filename):
        # this method will be implemented when a data sample is available
        raise NotImplementedError


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from valplat.util.path import TEST_DATA_PATH
    from valplat.util.gui import add_slider
    from valplat.util.log import initialize_logging

    def refresh(frame):
        t_master = scan_list[stream_names[0]][0][int(frame)]
        for (key, (t, scan)), g_i in zip(scan_list.items(), g):
            frame = search_sorted_nearest(t_master, t)
            scan = scan[int(frame)]
            g_i.set_data(*scan["x"].T)

    initialize_logging()
    test_data = TEST_DATA_PATH / "livox_multi_raw_sample.lvx"
    stream_names = LvxReader().get_stream_names(test_data, print_info=True)
    scan_list = {
        key: LvxReader(stream_name=key).extract(test_data, 0, 10)
        for key in stream_names
    }

    _, span = plt.figure(figsize=(12, 7.5)), 5
    grid = (span + 1, 1)
    axis_range = (60, 30)
    axes = [
        plt.subplot2grid(grid, (0, 0), rowspan=span),
        plt.subplot2grid(grid, (span, 0), colspan=grid[1]),
    ]
    g = [axes[0].plot([], ".", ms=1, alpha=0.2)[0] for key in stream_names]
    add_slider(axes[-1], refresh, valmax=len(scan_list[stream_names[0]][0]) - 1)
    axes[0].axis([0, 100, -10, 10])
    plt.show()
