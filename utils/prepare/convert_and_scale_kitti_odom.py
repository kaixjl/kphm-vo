from typing import Callable, List, Tuple, Union
import imageio
import skimage
import skimage.color
import cv2 as cv
import numpy as np
import os
import shutil
from datasets import KittiOdometryLoader
from ..misc import mkdir_if_not_exist

def convert_and_scale_kitti_odom(dataset_dir, to_dir, img_height, img_width, sequences: List[int], cut=None):
    '''
    scale, and convert to jpg, save to a new directory. (means won't modify the original files)
    '''
    if len(sequences) == 0:
        return
    kitti_odom_loader = KittiOdometryLoader(dataset_dir, img_height=img_height, img_width=img_width, sequences=sequences, cut=cut)
    mkdir_if_not_exist(to_dir)
    sequences_dir = os.path.join(to_dir, "sequences")
    pose_dir = os.path.join(to_dir, "poses")
    mkdir_if_not_exist(sequences_dir)
    mkdir_if_not_exist(pose_dir)
    for seq_id in kitti_odom_loader.sequences:
        print("Processing sequence {}...".format(seq_id))
        print("Copy poses file and convert calib.")
        shutil.copyfile(os.path.join(dataset_dir, "poses", "{:02d}.txt".format(seq_id)), os.path.join(pose_dir, "{:02d}.txt".format(seq_id))) # copy pose file
        seq_dir = os.path.join(sequences_dir, "{:02d}".format(seq_id))
        mkdir_if_not_exist(seq_dir)
        dataset_seq_dir = os.path.join(dataset_dir, "sequences", "{:02d}".format(seq_id))
        # shutil.copyfile(os.path.join(dataset_seq_dir, "calib.txt"), os.path.join(seq_dir, "calib.txt"))
        # shutil.copyfile(os.path.join(dataset_seq_dir, "times.txt"), os.path.join(seq_dir, "times.txt")) # copy times file
        with open(os.path.join(seq_dir, "calib.txt"), 'w') as f:
            for camera_id in range(4): # scale intrinsics
                p = kitti_odom_loader.load_projection_matrix(seq_id, camera_id) # type: np.ndarray
                p = np.reshape(p, (-1,))
                f.write('P{}:'.format(camera_id))
                for i in range(12):
                    f.write(' {:.12e}'.format(p[i]))
                f.write('\n')

        print("Convert frames into jpg format and scale")
        for camera_id in [2,3]:
            print("Camera {}".format(camera_id))
            img_dir = os.path.join(seq_dir, "image_{}".format(camera_id))
            mkdir_if_not_exist(img_dir)
            frames = kitti_odom_loader.frames_in_sequence(seq_id)
            for frame_id in frames: # convert frames into jpg and scale
                print("{}/{}\r".format(frame_id + 1, len(frames)), end="")
                img = kitti_odom_loader.load_image(seq_id, frame_id, camera_id)
                imageio.imsave(os.path.join(img_dir, "{:010d}.jpg".format(frame_id)), skimage.img_as_ubyte(img))
            print("{}/{} finished.".format(len(frames), len(frames)))