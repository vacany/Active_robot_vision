## Toolbox for FrontCam and Livox geometric calibration


### Installation

Install the `lfccalib` (Livox FrontCam CALIBrattion) package:
```
pip install -e /path/to/livox_frontcam_calibration
```

The package can be uninstalled with:
```
pip uninstall lfccalib
```

### Structure

```
lfccalib 
|-- config.py         # Contain calibration parameters
|-- calibration_2d.py # Function for image undistorsion
|-- calibration_3d.py # Function for 3D transformation and projection to FrontCam plane
```

### Testing the toolbox

You can test the installation by running
```
python /path/to/livox_frontcam_calibration/test_calibration.py
```

The above script loads a Livox point cloud and an image from the FrontCam, un-distorts the image and projects the point cloud on the image. A visualization of the result is saved in
```
 /path/to/livox_frontcam_calibration/result.png
```
which can be compared to the result saved in 
```
/path/to/livox_frontcam_calibration/test_data/expected_result.png
```
