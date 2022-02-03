# Python Package Installation

Python version: python 3

Use following command to install python package or install manually referring to requirements.txt.

```
> pip install -r requirements.txt
```

# Preparation

## Kitti Odometry Dataset

We use kitti odometry dataset. Kitti Odometry dataset directory should be organized as following.

```
<Kitti Odometry Dataset Directory>
    |- poses
    |   |- <pose files>
    |- sequences
    |- <sequences folders>
        |- image_2
        |- <other folders>
        |- calib.txt
        |- <other files>
```

In this repo, default directory storing kitti odometry dataset is `dataset/kitti_odom/original`.

## Preprocessing

Specify `ori_dataset_dir` (original kitti odometry dataset directory, storing original dataset), `dataset_dir` (output directory, storing preprocessed dataset), `dataset_related_dir` (related directory, storing keypoint heatmap) and other prepare and data preprocessing section fields in params.py. Then use following command to preprocessing the dataset and generate splits.

```
> python prepare.py --gen_preprocessed_dataset --remove_static --gen_heatmap --gen_split
```

Generated split storing in `splits/<split_tag>`, where `split_tag` can be specified in params.py.

# Training

Use following command to training.

```
> python train.py -u s1 -c s1_tag --cudas 4 --epoch_size 60 --batch_size 3
```

in which, `-u` indicate use `s1_param` in params.py, `-c` specify the training tag that can be used in testing and other. Except `-u`, other switches can be specifed in params.py.

Models are saved in `records/s1_tag`, tensorboard files are saved in `runs/s1_tag`.

# Testing

Use following command to test.

```
> python test_pose.py -c s1_tag --cudas 4
```

in which, `-c` indicate training tag to be tested, here is the same with it in training.

# Others

`show_sample.py`: generate sample image, specified by test_seq_id, test_frame_idx, test_version in params.py, using saving models in training.

`archive_record.py`: saving test result into `archive` directory.

`result_to_xlsx.py`: generate `xlsx` file using result in `archive` directory.
