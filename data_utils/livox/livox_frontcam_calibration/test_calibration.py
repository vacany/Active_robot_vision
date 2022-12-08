import os
import cv2
import numpy as np
from lfccalib import config
import lfccalib.calibration_2d as c2d
import lfccalib.calibration_3d as c3d
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # --- Load raw data
    print("Load test data...")
    image = cv2.imread(
        os.path.dirname(os.path.realpath(__file__)) + \
        "/test_data/22-05-10_08-27-48_HondaJazz_F6021_CZ_D_OC_UR_RIGHT_OV_0001.png"
    )
    pc = np.load(
        os.path.dirname(os.path.realpath(__file__)) + \
        "/test_data/jazz_camera_2022-05-10-10-28-23/time_0.0000.npz"
    )["point_cloud"]

    # --- Livox Lidars Calibration
    print("Calibrate Livox Lidars...")
    # The "factory" calibration was removed beforehand in the test data so the following line should not be applied.
    # However, on "normal" recording, please uncomment the following line.
    # `pc = c3d.calibrate_livox_point_clouds(pc, config.get_honda_jazz_factory_calibration())`
    pc = c3d.calibrate_livox_point_clouds(
        pc, config.get_honda_jazz_livox_fine_tuned_calibration()
    )

    # --- Camera undistorsion
    print("Compute undistorsion map for OV FrontCam...")
    map = c2d.get_undistortion_map(
        config.get_honda_jazz_undistortion_param(), image.shape[:2]
    )
    # For faster processing the map above could be saved on disk
    print("Undistort image...")
    image = c2d.undistort_image(image, map)

    # --- Projection Livox + Camera
    print("Project point cloud on camera plane...")
    cami, came = config.get_honda_jazz_intrinsic_extrinsic()
    pc_proj = c3d.project_points_to_camera(pc, cami, came, image.shape[:2])

    # --- Display result
    print("Show result (limiting depth at 50 meters)...")
    # Limit depth
    where = pc_proj[:, 2] < 50
    pc_proj = pc_proj[where]
    # Display result
    fig, ax = plt.subplots(1, 1) 
    ax.imshow(image)
    ax.scatter(pc_proj[:, 0], pc_proj[:, 1], c=pc_proj[:, 3], s=1)
    ax.axis("off")
    filename = os.path.dirname(os.path.realpath(__file__)) + "/result.png"
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    print(f"Saved results in {filename}.")
