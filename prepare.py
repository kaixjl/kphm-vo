import argparse
from typing import Callable, List, Tuple, Union
import os
from datasets import KittiOdometryLoader
from utils.prepare import *
from params import DatasetNameEnum, get_param

param = get_param()

def parse_arg():
    global param

    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_dataset_dir",   type=str, help="where the dataset is stored")
    parser.add_argument("--dataset_dir",   type=str, help="where to output the converted dataset")
    parser.add_argument("--output_dir",     type=str, help="where to output the prepared data other than converted dataset.")
    parser.add_argument("--seq_length",    type=int, help="length of each training sequence")
    parser.add_argument("--interval",    type=int, help="sample interval between each subsequence")
    parser.add_argument("--img_height",    type=int,   help="image height")
    parser.add_argument("--img_width",     type=int,   help="image width")
    parser.add_argument("--heatmap_radius",     type=int,   help="keypoint radius when generating heatmap")
    parser.add_argument("--remove_static", help="remove static frames from kitti raw data", action='store_true')
    parser.add_argument("--gen_preprocessed_dataset", help="convert dataset into jpg format and scale.", action='store_true')
    parser.add_argument("--gen_heatmap", help="generate keypoint heatmap.", action='store_true')
    parser.add_argument("--gen_split", help="generate split of train, val, test.", action='store_true')
    parser.add_argument("--split_tag",     type=str, help="split tag")
    parser.add_argument("--use_config", "-u", nargs="+", type=str, help="specify which config to be used.")
    args0 = parser.parse_args()

    param = get_param(args0.use_config)

    def choose(a, b):
        return a if a is not None else b

    param.ori_dataset_dir = choose(args0.ori_dataset_dir, param.ori_dataset_dir)
    param.dataset_dir = choose(args0.dataset_dir, param.dataset_dir)
    param.dataset_related_dir = choose(args0.output_dir, param.dataset_related_dir)
    param.seq_length = choose(args0.seq_length, param.seq_length)
    param.interval = choose(args0.interval, param.interval)
    param.img_height = choose(args0.img_height, param.img_height)
    param.img_width = choose(args0.img_width, param.img_width)
    param.heatmap_radius = choose(args0.heatmap_radius, param.heatmap_radius)
    param.remove_static = args0.remove_static
    param.split_tag = choose(args0.split_tag, param.split_tag)

    return args0

def main():
    args0 = parse_arg()

    if args0.gen_preprocessed_dataset:
        if not os.path.exists(param.dataset_dir):
            os.makedirs(param.dataset_dir)
        if param.dataset_name == DatasetNameEnum.KITTI_ODOM:
            convert_and_scale_kitti_odom(param.ori_dataset_dir, param.dataset_dir, param.img_height, param.img_width, sequences=param.train_sequence + param.valid_sequence + param.test_sequence) # convert images into jpg format and scale


    if args0.gen_split:
        if param.dataset_name == DatasetNameEnum.KITTI_ODOM:
            dataset_loader_train = KittiOdometryLoader(param.dataset_dir, seq_length=param.seq_length, sequences=param.train_sequence, remove_static=param.remove_static)
            dataset_loader_valid = KittiOdometryLoader(param.dataset_dir, seq_length=param.seq_length, sequences=param.valid_sequence, remove_static=param.remove_static)

        path_train_split = os.path.join("splits", "train.txt") if param.split_tag is None else os.path.join("splits", param.split_tag, "train.txt")
        path_val_split = os.path.join("splits", "val.txt") if param.split_tag is None else os.path.join("splits", param.split_tag, "val.txt")
        path_note = os.path.join("splits", "note.txt") if param.split_tag is None else os.path.join("splits", param.split_tag, "note.txt")

        if not os.path.exists(os.path.dirname(path_train_split)):
            os.makedirs(os.path.dirname(path_train_split))
        if not os.path.exists(os.path.dirname(path_val_split)):
            os.makedirs(os.path.dirname(path_val_split))

        gen_split(param.dataset_dir, path_train_split, dataset_loader_train, param.interval)
        gen_split(param.dataset_dir, path_val_split, dataset_loader_valid, param.interval)

        with open(path_note, "w") as f:
            f.write("train {}\nval {}\ntest {}".format(" ".join([str(i) for i in param.train_sequence]), " ".join([str(i) for i in param.valid_sequence]), " ".join([str(i) for i in param.test_sequence])))

    if args0.gen_heatmap:
        if not os.path.exists(param.dataset_related_dir):
            os.makedirs(param.dataset_related_dir)
        if param.dataset_name == DatasetNameEnum.KITTI_ODOM:
            gen_heatmap_kitti_odom(param.dataset_dir, param.dataset_related_dir, param.img_height, param.img_width, sequences=param.train_sequence + param.valid_sequence + param.test_sequence, radius=param.heatmap_radius)

if __name__=="__main__":
    main()

