import os
import glob
import lfccalib
import numpy as np
from .pc_dataset import PCDataset


class LivoxDataset(PCDataset):
    def __init__(
        self,
        root_dir,
        phase="val",
        which_lidar="all",
        voxel_size=[0.1, 0.1, 0.1],
    ):

        super().__init__(
            voxel_size=voxel_size,
        )
        self.phase = phase
        self.root_dir = root_dir
        self.local_dir = '/root/local_storage/livox_sim_data_npz/'
        os.makedirs(self.local_dir, exist_ok=True)
        self.which_lidar = which_lidar
        self.files = np.sort(glob.glob(os.path.join(root_dir, "points", "*.txt")))

        ind_split = len(self.files) * 9 // 10
        if self.phase == "train":
            self.files = self.files[:ind_split]
        elif self.phase == "val":
            self.files = self.files[ind_split:]
        else:
            raise ValueError(f"Phase {self.phase} not implemented.")

    def __len__(self):
        return len(self.files)

    def load_pc(self, index):
        # Load point cloud
        npz_files = self.local_dir + self.files[index][len(os.path.join(self.root_dir, "points")):-3] + 'npz'
        if os.path.exists(npz_files):
            data = np.load(npz_files)["data"]
        else:
            data = np.loadtxt(self.files[index], delimiter=",")
            np.savez(npz_files, data=data)
        # Concatenate xyz coords and
        # indicator function of tele-15 (6th lidar, other are horizon lidars)
        pc = np.concatenate((data[:, :3], data[:, -1:] == 6), axis=1)

        labels = data[:, 4]

        # labels[labels > 1] = 2 # Consider only cars

        # Filter
        if self.which_lidar == "all":
            filter = pc[:, -1] > -1
            # Leave indicator function of lidar type as feature
        elif self.which_lidar == "tele-15":
            filter = pc[:, -1] == 1
            pc[:, -1] = 1  # Constant feature
        elif self.which_lidar == "horizon":
            filter = pc[:, -1] == 0
            pc[:, -1] = 1  # Constant feature
        else:
            raise ValueError(f"Unknown lidar: {self.which_lidar}")

        return pc[filter], self.files[index] # labels[filter]


class ValeoLivoxDataset(PCDataset):

    def __init__(
        self,
        root_dir,
        which_lidar="all",
        feats="intensity+lidar_type",
        voxel_size=[0.1, 0.1, 0.1],
    ):

        super().__init__(
            voxel_size=voxel_size,
        )
        self.root_dir = root_dir
        self.which_lidar = which_lidar
        self.feats = feats.split("+")
        self.files = np.sort(glob.glob(os.path.join(root_dir, "**/*.npz")))
        print(f"Found {len(self.files)} npz files.")

    def __len__(self):
        return len(self.files)

    def load_pc(self, index):
        # Load point cloud
        pc = np.load(self.files[index])["point_cloud"]

        # Calibrate
        pc = lfccalib.calibration_3d.calibrate_livox_point_clouds(
            pc, lfccalib.config.get_honda_jazz_factory_calibration()
        )
        pc = lfccalib.calibration_3d.calibrate_livox_point_clouds(
            pc, lfccalib.config.get_honda_jazz_livox_fine_tuned_calibration()
        )

        # Restrict to desired features
        new_pc = [pc[:, :3]]
        ind_lidar_type = None
        for i, f in enumerate(self.feats):
            if f == "intensity":
                new_pc.append(pc[:, 3:4])
            elif f == "lidar_type":
                # Tele-15 (index: 2) and Horizon (index: 1 or 3)
                new_pc.append(pc[:, 4:5])
                ind_lidar_type = 3 + i
            elif f == "height":
                new_pc.append(pc[:, 2:3])
            else:
                raise ValueError("Unknow feature")
        pc = np.concatenate(new_pc, axis=1)
        # pc[:, 3] = 1.
        # pc = np.concatenate((pc[:, :3], pc[:, 4:5]), axis=-1)

        # Filter with respect to livox type
        if self.which_lidar == "all":
            filter = pc[:, ind_lidar_type] > 0
            pc[:, ind_lidar_type] = (pc[:, ind_lidar_type] == 2) # 1 for tele-15 ; 0 for horizon
        elif self.which_lidar == "tele-15":
            filter = pc[:, ind_lidar_type] == 2
            pc[:, ind_lidar_type] = 1  # Constant feature
        elif self.which_lidar == "horizon":
            filter = pc[:, ind_lidar_type] != 2
            pc[:, ind_lidar_type] = 1  # Constant feature
        else:
            raise ValueError(f"Unknow lidar: {self.which_lidar}")

        # return pc[filter, :4], self.files[index][len(self.root_dir):]
        return pc[filter], self.files[index][len(self.root_dir):]

