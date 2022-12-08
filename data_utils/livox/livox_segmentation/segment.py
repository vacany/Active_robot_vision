import os
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader

from models.minkunet import MinkUNet34
from models.pointnet import MiniPointNet
from utils.predictor import Predictor

from datasets.livox_dataset import ValeoLivoxDataset
from datasets.pc_dataset import collate_pointclouds


# BGR - SemanticKitti color code
colormap_15 = {
    0: [0, 0, 0],  # unknown
    1: [245, 150, 100],  # car
    2: [180, 30, 80],  # truck
    3: [250, 80, 100],  # bus
    4: [245, 230, 100],  # bicycle
    5: [150, 60, 30],  # motorbike
    6: [30, 30, 255],  # pedestrian
    7: [0, 0, 255],  # dog
    8: [255, 255, 255],  # road
    9: [80, 240, 150],  # ground
    10: [0, 200, 255],  # building
    11: [50, 120, 255],  # fence
    12: [0, 175, 0],  # tree
    13: [150, 240, 255],  # pole
    14: [75, 0, 175],  # greenbelt
}
colormap_3 = {
    0: [0,  0,  0], # unknown
    1: [255, 0, 0], #
    2: [0, 0, 255], #
}


if __name__ == "__main__":

    # ---
    parser = argparse.ArgumentParser(description="Inference on Valeo Livox data")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file",
        required=True,
    )
    parser.add_argument(
        "--path_livox_data",
        type=str,
        help="Path to livox data",
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path where to save predictions"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to load data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--color",
        action='store_true',
        help="Save prediction in txt file with colormap"
    )
    args = parser.parse_args()

    # --- Load config file for experiment
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # --- Build network
    if config["model"]["arch"] == "minkunet34":
        Model = MinkUNet34
    elif config["model"]["arch"] == "mini-pointnet":
        Model = MiniPointNet
    else:
        raise ValueError("Unkwown architecture")
    net = Model(
        in_channels=config["model"]["in_channels"],
        num_classes=config["model"]["nb_class"],
    )

    # --- Dataloader
    val_dataset = ValeoLivoxDataset( # LivoxDataset( #
        args.path_livox_data,
        feats=config["dataloader"]["feats"],
        which_lidar=config["dataloader"]["which_lidar"],
        voxel_size=config["dataloader"]["voxelization"],
    )
    loader_val = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=collate_pointclouds,
    )

    # Whether to save result for visualisation or in npz file
    if args.color:
        if config["model"]["nb_class"] == 15:
            colormap = colormap_15
        elif config["model"]["nb_class"] == 3:
            colormap = colormap_3
        colormap = np.array(
            [
                np.array(colormap[k])[::-1] / 255.
                for k in range(len(colormap))
            ]
        )
    else:
        colormap = None

    # --- Evaluation
    mng = Predictor(
        net,
        loader_val,
        args.ckpt_path,
        args.out_dir,
        args.device,
        colormap=colormap,
    )
    mng.eval()
