import numpy as np
import os
import glob
import pandas as pd

from scipy.spatial.transform import Rotation
from pathlib import Path
from pyarrow import feather

from timespace.box import get_point_mask


argo_cls_map_dict = {"ANIMAL" : 30,
           "ARTICULATED_BUS" : 23,
           "BICYCLE" : 7,
           "BICYCLIST" : 15,
           "BOLLARD" : 3,
           "BOX_TRUCK" : 11,
           "BUS" : 10,
           "CONSTRUCTION_BARREL" : 5,
           "CONSTRUCTION_CONE" : 4,
           "DOG" : 19,
           "LARGE_VEHICLE" : 8,
           "MESSAGE_BOARD_TRAILER" : 24,
           "MOBILE_PEDESTRIAN_CROSSING_SIGN" : 25,
           "MOTORCYCLE" : 14,
           "MOTORCYCLIST" : 18,
           "OFFICIAL_SIGNALER" : 28,
           "PEDESTRIAN" : 2,
           "RAILED_VEHICLE" : 1,
           "REGULAR_VEHICLE" : 27,
           "SCHOOL_BUS" : 20,
           "SIGN" : 12,
           "STOP_SIGN" : 6,
           "STROLLER" : 22,
           "TRAFFIC_LIGHT_TRAILER" : 29,
           "TRUCK" : 13,
           "TRUCK_CAB" : 17,
           "VEHICULAR_TRAILER" : 16,
           "WHEELCHAIR" : 26,
           "WHEELED_DEVICE" : 9,
           "WHEELED_RIDER" : 21,
            }

dyn_cls_map_dict = {"ANIMAL" : 1,
           "ARTICULATED_BUS" : 1,
           "BICYCLE" : 0,
           "BICYCLIST" : 1,
           "BOLLARD" : 0,
           "BOX_TRUCK" : 0,
           "BUS" : 1,
           "CONSTRUCTION_BARREL" : 0,
           "CONSTRUCTION_CONE" : 0,
           "DOG" : 1,
           "LARGE_VEHICLE" : 1,
           "MESSAGE_BOARD_TRAILER" : 0,
           "MOBILE_PEDESTRIAN_CROSSING_SIGN" : 0,
           "MOTORCYCLE" : 0,
           "MOTORCYCLIST" : 1,
           "OFFICIAL_SIGNALER" : 1,
           "PEDESTRIAN" : 1,
           "RAILED_VEHICLE" : 1,
           "REGULAR_VEHICLE" : 1,
           "SCHOOL_BUS" : 1,
           "SIGN" : 0,
           "STOP_SIGN" : 0,
           "STROLLER" : 1,
           "TRAFFIC_LIGHT_TRAILER" : 0,
           "TRUCK" : 1,
           "TRUCK_CAB" : 1,
           "VEHICULAR_TRAILER" : 0,
           "WHEELCHAIR" : 1,
           "WHEELED_DEVICE" : 0,
           "WHEELED_RIDER" : 1}



def read_feather(path: Path) -> pd.DataFrame:
    """Read Apache Feather data from a .feather file.

    AV2 uses .feather to serialize much of its data. This function handles the deserialization
    process and returns a `pandas` DataFrame with rows corresponding to the records and the
    columns corresponding to the record attributes.

    Args:
        path: Source data file (e.g., 'lidar.feather', 'calibration.feather', etc.)
        columns: Tuple of columns to load for the given record. Defaults to None.

    Returns:
        (N,len(columns)) Apache Feather data represented as a `pandas` DataFrame.
    """
    data: pd.DataFrame = feather.read_feather(path)
    return data

def quat_to_mat(quat_wxyz) -> np.array:
    """Convert a quaternion to a 3D rotation matrix.

    NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
        we use the scalar FIRST convention.

    Args:
        quat_wxyz: (...,4) array of quaternions in scalar first order.

    Returns:
        (...,3,3) 3D rotation matrix.
    """
    # Convert quaternion from scalar first to scalar last.
    quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
    mat = Rotation.from_quat(quat_xyzw).as_matrix()
    return mat

def find_nearest_timestamps(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def read_lidar(lidar_feather_path):
    lidar = read_feather(lidar_feather_path)

    xyz = lidar.loc[:, ["x", "y", "z"]].to_numpy().astype(float)
    intensity = lidar.loc[:, ["intensity"]].to_numpy().squeeze() / 255
    xyzi = np.concatenate((xyz, intensity[:, np.newaxis]), axis=1)

    laser_number = lidar.loc[:, ["laser_number"]].to_numpy().squeeze()
    offset_ns = lidar.loc[:, ["offset_ns"]].to_numpy().squeeze()

    upper_laser_idx = laser_number <= 32    # might be usable later
    lower_laser_idx = laser_number > 32

    return xyzi

def read_lidar_poses(av2_seq_path):
    # calibration = read_feather(av2_seq_path + '/calibration/egovehicle_SE3_sensor.feather')
    pose = read_feather(av2_seq_path + '/city_SE3_egovehicle.feather')
    all_pose_stamps = pose.loc[:, "timestamp_ns"].to_numpy()
    lidar_paths = sorted(glob.glob(av2_seq_path + '/sensors/lidar/*.feather'))
    # unpack poses
    poses = []
    for lidar_idx in range(len(lidar_paths)):
        cur_timestamp = int(os.path.basename(lidar_paths[lidar_idx]).split('.')[0])
        nearest_pose_idx = find_nearest_timestamps(all_pose_stamps, cur_timestamp)
        cur_pose = pose.loc[nearest_pose_idx]
        rot_mat = Rotation.from_quat([cur_pose['qx'], cur_pose['qy'], cur_pose['qz'], cur_pose['qw']]).as_matrix()
        translation = np.array((cur_pose['tx_m'], cur_pose['ty_m'], cur_pose['tz_m']))
        T = np.eye(4)
        T[:3, :3] = rot_mat
        T[:3, 3] = translation
        poses.append(T)

    poses = np.stack(poses)

    return poses

def read_annotation_file(av2_seq_path):
    annotations_feather_path = av2_seq_path + '/annotations.feather'  # IF train
    annotation_data = read_feather(annotations_feather_path)
    rotation = quat_to_mat(annotation_data.loc[:, ["qw", "qx", "qy", "qz"]].to_numpy())
    translation_m = annotation_data.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy()
    length_m = annotation_data.loc[:, "length_m"].to_numpy()
    width_m = annotation_data.loc[:, "width_m"].to_numpy()
    height_m = annotation_data.loc[:, "height_m"].to_numpy()
    category = annotation_data.loc[:, "category"].to_numpy()
    timestamp_ns = annotation_data.loc[:, "timestamp_ns"].to_numpy()
    id = annotation_data.loc[:, 'track_uuid'].to_numpy()

    track_mapping = {i: j for i, j in zip(np.unique(id), range(len(np.unique(id))))}
    cls_mapping = argo_cls_map_dict

    anno_list = []  # x,y,z,l,w,h, yaw, category, uuid, timestamp, nbr_interior_points

    for obj_idx in range(len(annotation_data)):
        x, y, z = translation_m[obj_idx]
        l, w, h = length_m[obj_idx], width_m[obj_idx], height_m[obj_idx]
        yaw = Rotation.from_matrix(rotation[obj_idx]).as_euler('xyz')[2]
        clz = cls_mapping[category[obj_idx]]
        dyn_prior = dyn_cls_map_dict[category[obj_idx]]
        uuid = track_mapping[id[obj_idx]]
        timestamp_object = timestamp_ns[obj_idx]
        interior_pts = annotation_data.loc[obj_idx, 'num_interior_pts']

        one_object = np.array((x, y, z, l, w, h, yaw, clz, uuid, timestamp_object, interior_pts, dyn_prior))
        anno_list.append(one_object)

    annotation_data = np.stack(anno_list)
    annotation_dataframe = pd.DataFrame(annotation_data,
                                        columns=['x', 'y', 'z', 'l', 'w', 'h', 'yaw', 'clz', 'id', 'ts_ns', 'nbr_pts', 'dyn_prior'])


    # Transfered annotations
    lidar_paths = sorted(glob.glob(av2_seq_path + '/sensors/lidar/*.feather'))

    for folder in ['seg_labels', 'id_mask']:
        os.makedirs(av2_seq_path + '/' + folder, exist_ok=True)

    for lidar_idx in range(len(lidar_paths)):
        print(lidar_idx)
        cur_timestamp = int(os.path.basename(lidar_paths[lidar_idx]).split('.')[0])

        xyzi = read_lidar(lidar_paths[lidar_idx])
        cur_boxes = annotation_data[annotation_data[..., 9] == cur_timestamp]

        id_mask = np.zeros(xyzi.shape[0], dtype=int)
        seg_mask =  np.zeros(xyzi.shape[0], dtype=int)

        for box in cur_boxes:

            mask = get_point_mask(xyzi[:,:3], box[:7])
            id_mask[mask] = box[8]
            seg_mask[mask] = box[7]

        np.save(f"{av2_seq_path}/id_mask/{lidar_idx:06d}.npy", id_mask)
        np.save(f"{av2_seq_path}/seg_labels/{lidar_idx:06d}.npy", seg_mask)


    annotation_dataframe.to_csv(f"{av2_seq_path}/processed_annotation.csv")
    # readMe
    info_str = "annotation columns: 'x', 'y', 'z', 'l', 'w', 'h', 'yaw', 'clz', 'id', 'ts_ns', 'nbr_pts', 'dyn_prior' \n" \
               "poses in poses.npy are associated to lidar streams \n" \
               "seg_labels are segmentation labels constructed from bounding boxes with the dictionary in argoverse2.py \n" \
               "id_mask are reconstructed to per-point format with integers instead of tokens"

    with open(f"{av2_seq_path}/processed_info.txt", 'w') as f:
        f.writelines(info_str)
        f.close()

    return annotation_dataframe

def prepare_sequence(av2_seq_path):
    print("Preparing the sequence ---------")
    print('poses and lidars')
    poses = read_lidar_poses(av2_seq_path)
    np.save(av2_seq_path + '/poses.npy', poses)

    os.makedirs(f"{av2_seq_path}/lidar", exist_ok=True)

    for idx, file in enumerate(sorted(glob.glob(f"{av2_seq_path}/sensors/lidar/*.feather"))):
        xyzi = read_lidar(file)
        np.save(f"{av2_seq_path}/lidar/{idx:06d}.npy", xyzi)

    if '/train/' in av2_seq_path or '/val/' in av2_seq_path:
        print('annotation')
        annotation = read_annotation_file(av2_seq_path)
    print('done')

def get_ego_bbox_argo():
  # https://www.edmunds.com/ford/fusion-hybrid/2020/features-specs/

  l = 191.8 * 0.0254    # in inches
  w = 83.5 * 0.0254
  h = 58.0 * 0.0254
  x, y, z = 0, 0, h / 2 - 0.2
  angle = 0
  EGO_BBOX = np.array((x, y, z, l, w, h, angle))

  return EGO_BBOX

if __name__ == '__main__':
    av2_seq_path = '/mnt/personal/vacekpa2/data/argoverse2/sensor/train/8606d399-57d4-3ae9-938b-db7b8fb7ef8c'
    prepare_sequence(av2_seq_path)
