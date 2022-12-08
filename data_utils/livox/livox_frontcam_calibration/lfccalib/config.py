import numpy as np


def get_honda_jazz_undistortion_param():
    """
    FrontCam OV parameters for correcting distortion
    """
    return [
        [474.90, 915.79],
        [
            -0.0006970405762,
            -0.0008928639648,
            0.001159733887,
            -0.001202057031,
            0.001137053711,
            -0.001015361523,
        ],
    ]


def get_honda_jazz_intrinsic_extrinsic():
    """
    FrontCam OV intrinsic and extrinsic parameters for 3D to 2D
    perspective projection
    Unit: mm.
    """
    cami = [
        [1501.34595, 0.00000000, 915.795166016],
        [0.0, 1501.34595, 469.096923828],
        [0.0, 0.0, 1.0],
    ]
    came = [
        [
            -0.0188622920720534,
            -0.999818951821792,
            -0.00250549710868142,
            17.9490187682126,
        ],
        [
            0.0668119489564397,
            0.00123989484481735,
            -0.997764815042811,
            -537.207401182635,
        ],
        [0.997587278093717, -0.0189875285054917, 0.0667764655160973, -889.38467319197],
        [0.000000000, 0.000000000, 0.000000000, 1.000000000],
    ]
    return np.array(cami), np.array(came)


def get_honda_jazz_factory_calibration():
    """
    Get default calibration matrices for each livox lidar as stored in lvx files.
    The parameters should be used to remove the default calibration and apply the fine tuned parameters in:
    get_honda_jazz_livox_fine_tuned_calibration().
    **Note:** that we provide directly the inverse transformation, ie the transformation matrices that remove the
    default calibration!
    """

    index_left = 1
    angle_left = 25.600000 / 180 * np.pi
    c, s = np.cos(angle_left), np.sin(angle_left)
    calib_left = np.array(
        [
            [c, -s, 0, -0.05000],
            [s, c, 0, 0.150000],
            [0, 0, 1, 1.890000],
            [0, 0, 0, 1.0],
        ]
    )

    index_center = 2
    angle_center = np.pi
    c, s = np.cos(angle_center), np.sin(angle_center)
    calib_center = np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 1.920000],
            [0, 0, 0, 1.0],
        ]
    )

    index_right = 3
    angle_right = -25.600000 / 180 * np.pi
    c, s = np.cos(angle_right), np.sin(angle_right)
    calib_right = np.array(
        [
            [c, -s, 0, -0.050000],
            [s, c, 0, 0.150000],
            [0, 0, 1, 1.890000],
            [0, 0, 0, 1.0],
        ]
    )

    return {
        index_left: np.linalg.inv(calib_left)[:3],
        index_center: np.linalg.inv(calib_center)[:3],
        index_right: np.linalg.inv(calib_right)[:3],
    }


def get_honda_jazz_livox_fine_tuned_calibration():
    """
    Get refined calibration matrices for each livox lidar
    """

    # Horizon
    index_left = 1
    calib_left = np.array(
        [
            [0.903492611, -0.428544338, -0.007131091, -69.450251480 * 1e-3],
            [0.428558497, 0.903513890, 0.000515148, 200.478748341 * 1e-3],
            [0.006222276, -0.003521522, 0.999974441, -7.758251415 * 1e-3],
        ]
    )

    # Tele-15
    index_center = 2
    angle_center = np.pi
    c, s = np.cos(angle_center), np.sin(angle_center)
    calib_center = np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
        ]
    )

    # Horizon
    index_right = 3
    calib_right = np.array(
        [
            [0.901182403, 0.433432734, 0.002518224, -76.697714906 * 1e-3],
            [-0.433438631, 0.901179106, 0.002678048, -194.008566060 * 1e-3],
            [-0.001108618, -0.003504905, 0.999993243, -12.541884475 * 1e-3],
        ]
    )

    return {
        index_left: calib_left,
        index_center: calib_center,
        index_right: calib_right,
    }
