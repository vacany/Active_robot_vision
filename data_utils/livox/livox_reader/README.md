##Convert lvx files and save them to npz files


### Usage

```
python /path/to/livox_reader/lvx2npz.py -i /path/to/input_folder/ -o /path/to/output_folder/
```
The script will search all `lvx` files contained in `/path/to/input_folder/` and save the point 
clouds in `npz` files (one point cloud per `npz` file) in `output_folder`. 


### Output folder structure

Suppose we have a input folder named `/data/city/` containing 2 `lvx` files: `file1.lvx` and `file2.lvx`. 
We extract these data by typing:
```
python /path/to/livox_reader/lvx2npz.py -i /data/city/ -o /data/livox_npz/
```

The output folder will have the following structure:
```
/data/livox_npz/
 |
 |-- file1/
     |--> time_aaa.aa.npz # Point cloud captured at time aaa.aa
     |.
     |--> time_bbb.bb.npz # Point cloud captured at time bbb.bb
 |-- file2/
     |--> time_ccc.cc.npz # Point cloud captured at time ccc.cc
     |.
     |--> time_ddd.dd.npz # Point cloud captured at time ddd.dd
```

### File structure

The point cloud captured, e.g., at time `aaa.aa` above can be loaded in python by typing:
```
data = np.load("/data/livox_npz/file1/time_aaa.aa.npz")
point_cloud = data["point_cloud"]
timestamp = data["timestamp"]
```

The `point_cloud` is a numpy array of size `(n, 5)` where `n` is the number of points:  
- the first 3 columns `point_cloud[:, :3]` contain the (x, y, z) coordinates of each point; 
- the 4th column `point_cloud[:, 3]` is the intensity normalized between 0 and 1; 
- the 5th is `point_cloud[:, 4]` the lidar index ranging from 1 to 3 (on the data I have, 
1 is the Horizon on the left, 2 is the Tele-15, and 3 is the Horizon on the right).

The timestamp associated to the point cloud can be converted back to a datetime as follows:
```
import datetime as dt
timestamp = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=float(timestamp))
```
