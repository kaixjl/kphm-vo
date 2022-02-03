import math
from collections import OrderedDict
from typing import Any, Callable, List, Tuple, Union
import imageio
import skimage
import skimage.color
import cv2 as cv
import numpy as np
import os
from datasets import KittiOdometryLoader
from params import DatasetNameEnum, get_param
from ..misc import mkdir_if_not_exist
from .gen_heatmap import GenImgHeatmap

param = get_param()

def gen_heatmap_kitti_odom(dataset_dir, to_dir, img_height, img_width, sequences, radius):
    '''
    generate keypoint heatmap of each frame.
    '''
    HEATMAP_RADIUS = radius
    KEYPOINT = param.keypoint
    KEYPOINTS = {"SIFT": lambda: cv.SIFT_create(),
                 "SURF": lambda: cv.xfeatures2d.SURF_create(400)}
    HEATMAP_KERNEL_FUNC = param.heatmap_kernel_func
    HEATMAP_KERNEL_FUNCS = {"linear": lambda x: (HEATMAP_RADIUS-x)/HEATMAP_RADIUS,
                            "gaussian": lambda x: 1/(math.sqrt(2*math.pi)*HEATMAP_RADIUS/3)*math.exp(-x**2/(2*(HEATMAP_RADIUS/3)**2))}

    if len(sequences) == 0:
        return
    
    kitti_odom_loader = KittiOdometryLoader(dataset_dir, img_height=img_height, img_width=img_width, sequences=sequences)
    gen_img_heatmap = GenImgHeatmap(HEATMAP_RADIUS, HEATMAP_KERNEL_FUNCS[HEATMAP_KERNEL_FUNC])

    kp_detector = KEYPOINTS[KEYPOINT]()

    mkdir_if_not_exist(to_dir)
    if not os.path.exists(to_dir):
        raise Exception("{} not exists. Probably fail to create. You can create directories manually or check permissions.".format(to_dir))
    sequences_dir = os.path.join(to_dir, "kpheatmap")
    mkdir_if_not_exist(sequences_dir)
    
    # output description
    description = {"HEATMAP_RADIUS":HEATMAP_RADIUS,
                   "KEYPOINT":KEYPOINT,
                   "HEATMAP_KERNEL_FUNC":HEATMAP_KERNEL_FUNC,
                   }
                   
    if os.path.exists(os.path.join(to_dir, "description.txt")):
        with open(os.path.join(to_dir, "description.txt"), 'r') as f:
            content = f.read().splitlines()
            content = OrderedDict(list(map(lambda x: x.strip(), i.split(":"))) for i in content)
            content.update(description)
    else:
        content = description

    with open(os.path.join(to_dir, "description.txt"), 'w') as f:
        f.write("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), content.items())))

    print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), description.items())))

    for seq_id in kitti_odom_loader.sequences:
        print("Processing sequence {}...".format(seq_id))
        seq_dir = os.path.join(sequences_dir, "{:02d}".format(seq_id))
        mkdir_if_not_exist(seq_dir)

        print("Generate heatmap")
        for camera_id in [2,3]:
            print("Camera {}".format(camera_id))
            img_dir = os.path.join(seq_dir, "image_{}".format(camera_id))
            mkdir_if_not_exist(img_dir)
            frames = kitti_odom_loader.frames_in_sequence(seq_id)
            print(end="")
            for i, frame_id in enumerate(frames): # convert frames into jpg and scale
                # if frame_id % 10 == 0:
                print("\r{}/{}.".format(i + 1, len(frames)), flush=True, end="")
                img = kitti_odom_loader.load_image(seq_id, frame_id, camera_id)
                img = skimage.color.rgb2gray(img)
                img = skimage.img_as_ubyte(img)

                kps_xy = kp_detector.detect(img, None)
                kps_uv = [ (int(kp.pt[1]), int(kp.pt[0])) for kp in kps_xy ]

                heatmap = gen_img_heatmap(img_height, img_width, kps_uv)

                imageio.imsave(os.path.join(img_dir, "{:010d}.png".format(frame_id)), skimage.img_as_ubyte(heatmap))
            # } for i, frame_id in enumerate(frames)
            print("{}/{} finished.".format(len(frames), len(frames)), flush=True)
        # } for camera_id in [2,3]
    # } for seq_id in kitti_odom_loader.sequences
    pass