import os
import glob
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from data_utils.livox.valplat.adapters.livox.livox_reader import LvxReader, search_sorted_nearest


def convert_data(file, time, out_dir, apply_ego_trans):

    # ---
    name_out = os.path.splitext(os.path.basename(file))[0].replace(" ", "-")
    out_path = os.path.join(out_dir, name_out)
    try:
        os.makedirs(out_path, exist_ok=False)
    except FileExistsError:
        print(
            f"Error! Folder:\n  {out_path}\nalready exists. "
            f"Remove this folder manually if you want to re-extract the data.\n"
        )
        exit(1)

    # ---
    stream_names = LvxReader().get_stream_names(file, print_info=True)
    scan_list = {
        key: LvxReader(stream_name=key, apply_ego_trans=apply_ego_trans).extract(
            file,
            time[0],
            time[1],
            reset_recording_time_offset=False,
        )
        for key in stream_names
    }

    # ---
    for t_master in tqdm(scan_list[stream_names[0]][0]):
        # Extract point cloud
        xyz_concat = []
        for ind, (t, scan) in enumerate(scan_list.values()):
            scan = scan[search_sorted_nearest(t_master, t)]
            xyz = np.concatenate(
                (
                    scan["x"][:, 0:1],  # x
                    scan["x"][:, 1:2],  # y
                    scan["z"][:, None],  # z coord
                    scan["i"][:, None],  # Intensity
                    scan["layer_id"][:, None],  # Lidar index
                ),
                axis=1,
            )
            xyz_concat.append(xyz)
        xyz_concat = np.concatenate(xyz_concat, axis=0)
        # Filename & Save point cloud
        filename = f"time_{t_master:.4f}.npz"
        np.savez(
            os.path.join(out_path, filename), point_cloud=xyz_concat, timestamp=t_master
        )


if __name__ == "__main__":

    # ---
    print("\n\n" + "#" * 24)
    print("Convert lvx files to npz")
    print("#" * 24 + "\n\n")
    parser = ArgumentParser(
        description="Convert lvx data to npz without any processing of the data."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        type=str,
        help="Path to input directory containing lvx files.",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, type=str, help="Path to output directory."
    )
    args = parser.parse_args()

    # ---
    files = np.sort(glob.glob(args.input_dir + "/*.lvx"))
    print(f"Number of lvx files found: {len(files)}. List of files:")
    for f in files:
        print(" " * 3 + f)
    for i, f in enumerate(files):
        print(f"\n\nExtract lvx file {i + 1} out of {len(files)}:\n")
        convert_data(f, [-np.inf, np.inf], args.output_dir, False)
