## Livox semantic segmentation

### Basic usage

Perform semantic segmentation
```
python /path/to/livox_segmentation/segment.py \
    --config /path/to/config.yaml
    --path_livox_data /path/to/livox/npz/folder
    --ckpt_path /path/to/trained/model
    --out_dir /path/to/output
```

*Remark: The list of available config files and checkpoints is described below.*

The script `segment.py` will scan all the subfolders of `/path/to/livox/npz/folder` to find `npz` files and 
will save the result in `/path/to/output` in npz file as well.

If the input folder as structure
```
/path/to/livox/npz/folder
 |-- sequence1/
     |--> time_aaa.aa.npz 
     |.
     |--> time_bbb.bb.npz 
 |-- sequence2/
     |--> time_ccc.cc.npz 
     |.
     |--> time_ddd.dd.npz 
```
the output folder will have an identical structure, i.e., 
```
/path/to/output
 |-- sequence1/
     |--> time_aaa.aa_seg.npz 
     |.
     |--> time_bbb.bb_seg.npz 
 |-- sequence2/
     |--> time_ccc.cc_seg.npz 
     |.
     |--> time_ddd.dd_seg.npz 
```

The result of the segmentation in `time_aaa.aa_seg.npz` can be loaded as
```
segmentation = np.load("time_aaa.aa_seg.npz")["segmentation"]
``` 
The array `segmentation` has size (n, 5) where:
- the first 3 columns `segmentation[:, :3]` contain the (x, y, z) coordinates of each point; 
- the 4th column `segmentation[:, 3]` is the class index (from 0 to NB_CLASSES) of each point; 
- the 5th is `segmentation[:, 4]` is the associated class probability.


### Usage for visual inspection in, e.g., CloudCompare

Run
```
python /path/to/livox_segmentation/segment.py \
    --config /path/to/config.yaml
    --path_livox_data /path/to/livox/npz/folder
    --ckpt_path /path/to/trained/model
    --out_dir /path/to/output
    --color
```
The output folder will have a structure
```
/path/to/output
 |-- sequence1/
     |--> time_aaa.aa.txt
     |.
     |--> time_bbb.bb.txt
 |-- sequence2/
     |--> time_ccc.cc.txt
     |.
     |--> time_ddd.dd.txt
```
which can be loaded directly in CloudCompare. The first 3 columns in the txt file
contain the (x, y, z) coordinates of each point, and the last 3 columns the RGB color
(between 0 and 1) associated to each class.


### Trained models

#### Livox synthetic dataset

*Config file: `./configs/minkunet34_all_synthetic.yaml`*

*Checkpoint: `minkunet34_all_synthetic.pth`*

This training was performed using the synthetic dataset available at 
`https://livox-wiki-en.readthedocs.io/en/latest/data_summary/dataset.html` with the following classes:

| Class name | Index |
| ---------- | ----- |
| unknown    |  0    |
| car        |  1    |
| truck      |  2    | 
| bus        |  3    |
| bicycle    |  4    | 
| motorcyle  |  5    | 
| pedestrian |  6    | 
| dog        |  7    |
| road       |  8    |
| ground     |  9    |
| building   | 10    | 
| fence      | 11    | 
| tree       | 12    |
| pole       | 13    |
| greenbelt  | 14    |


#### Finetuning for "Car" on real data

*Config file: `./configs/minkunet34_car_finetune.yaml`*

*Checkpoint: `minkunet34_car_finetune.pth`*

| Class name | Index |
| ---------- | ----- |
| unknown    |  0    |
| car        |  1    |
| not car    |  2    | 


#### "Traffic Sign" trained on real data

*Config file: `./configs/miniPointnet_trafficSign.yaml`*

*Checkpoint: ``*

| Class name        | Index |
| ----------------- | ----- |
| unknown           |  0    |
| not traffic sign  |  1    |
| traffic sign      |  2    | 