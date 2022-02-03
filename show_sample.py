from typing import Iterable, Tuple
import os
import os.path
import pickle
import numpy as np
import torch
import argparse
import imageio
from datasets import *
from models.misc import create_networks
from utils.torchutils import pil_to_tensor, tensor_to_pil, sample, Rt_from_axisangle_and_trans, disp_to_depth, Reproject
from params import DatasetNameEnum, get_param

param = get_param()

def parse_arg():
    global param

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",   type=str, help="where the dataset is stored")
    parser.add_argument("--seq_id", type=str, help="seq id to be test")
    parser.add_argument("--frame_idx", type=int, help="frame id to be test")
    parser.add_argument("--config_name", "-c",   type=str, help="specify the tag of the running to be test")
    parser.add_argument("--not_cuda", help="use cpu but cuda.", action='store_true')
    parser.add_argument("--depth_only", help="only predict depth.", action='store_true')
    parser.add_argument("--archive",   type=str, help="specify archive tag if running archive")
    parser.add_argument("--root",   type=str, help="specify root dir")
    parser.add_argument("--use_config", "-u", nargs="+", type=str, help="specify which config to be used.")
    parser.add_argument("--cudas", type=str, help="specify cudas to be used. seperated with comma.")
    parser.add_argument("--network", type=int, help="specify network to be used.")
    parser.add_argument("--test_version", type=int, help="specify version to be test")
    args0 = parser.parse_args()

    param = get_param(args0.use_config)

    def choose(a, b):
        return a if a is not None else b

    param.dataset_dir = choose(args0.dataset_dir, param.dataset_dir)
    param.test_seq_id = choose(args0.seq_id, param.test_seq_id)
    param.test_frame_idx = choose(args0.frame_idx, param.test_frame_idx)
    param.config_name = choose(args0.config_name, param.config_name)
    param.not_cuda = args0.not_cuda
    param.cuda_visible_devices = choose(args0.cudas, param.cuda_visible_devices)
    param.test_version = choose(args0.test_version, param.test_version)

    if param.dataset_name == DatasetNameEnum.KITTI_ODOM:
        param.test_seq_id = int(param.test_seq_id)

    return args0

def main():
    args0 = parse_arg()
    show_sample(param.dataset_dir, param.test_seq_id, param.test_frame_idx, param.config_name, args0.depth_only, args0.archive, args0.root, args0.network, version=param.test_version)

def show_sample(dataset_dir, seq_id, frame_idx, config_name, depth_only=False, archive=None, root=None, network=None, version=0):
    '''
    frame_idx - 1 and frame_id will be used.
    '''
    # PARAMETERS
    CONFIG_NAME = config_name
    CUDA = not param.not_cuda
    CUDA_VISIBLE_DEVICES = param.cuda_visible_devices # int or str formatting comma-seperated-integer like "1,2,3,0" is acceptable
    NETWORK = param.network
    IMG_HEIGHT = param.img_height
    IMG_WIDTH = param.img_width
    DATASET_NAME = param.dataset_name
    DEPTH_MAX = param.depth_max
    DEPTH_MIN = param.depth_min
    
    # paths
    path_records_dir = os.path.join("records", CONFIG_NAME)
    if archive is not None:
        path_records_dir = os.path.join("archive", archive, path_records_dir)
    if root is not None:
        path_records_dir = os.path.join(root, path_records_dir)
    path_result_dir = os.path.join("result", CONFIG_NAME)

    with open(os.path.join(path_records_dir, "hyperparams.pickle"), 'rb') as f:
        hparams = pickle.load(f) # type: dict
        NETWORK = network if network is not None else (hparams["NETWORK"] if "NETWORK" in hparams.keys() else NETWORK)
        IMG_HEIGHT = hparams["IMG_HEIGHT"] if "IMG_HEIGHT" in hparams.keys() else IMG_HEIGHT
        IMG_WIDTH = hparams["IMG_WIDTH"] if "IMG_WIDTH" in hparams.keys() else IMG_WIDTH
        DATASET_NAME = hparams["DATASET_NAME"] if "DATASET_NAME" in hparams.keys() else DATASET_NAME
        DEPTH_MAX = hparams["DEPTH_MAX"] if "DEPTH_MAX" in hparams.keys() else DEPTH_MAX
        DEPTH_MIN = hparams["DEPTH_MIN"] if "DEPTH_MIN" in hparams.keys() else DEPTH_MIN
    
    if CUDA_VISIBLE_DEVICES is not None: # set CUDA_VISIBLE_DEVICES
        if type(CUDA_VISIBLE_DEVICES) is int:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
        elif type(CUDA_VISIBLE_DEVICES) is tuple or type(CUDA_VISIBLE_DEVICES) is list:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in CUDA_VISIBLE_DEVICES)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    TENSOR_DEVICE = torch.device("cuda") if CUDA and torch.cuda.is_available() else torch.device("cpu")
    

    # create Dataset and DataLoader object
    if DATASET_NAME == DatasetNameEnum.KITTI_ODOM:
        loader = KittiOdometryLoader(dataset_dir, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

    reproject = Reproject(IMG_HEIGHT, IMG_WIDTH, device=TENSOR_DEVICE)

    # create Network
    print("network")
    fullnet = create_networks(NETWORK, param, False, IMG_HEIGHT, IMG_WIDTH, False)
    if isinstance(fullnet, Iterable):
        fullnet = fullnet[-1]

    print("resume")
    fullnet.set_records_path(path_records_dir)

    # Iterative
    if depth_only:

        depthnet = fullnet.get_depthnet()
        depthnet.resume_models(version=version)
        depthnet.eval()
        depthnet.to(TENSOR_DEVICE)

        frame_ids = loader.frames_in_sequence(seq_id)
        frame_id2 = frame_ids[frame_idx]
        img1 = pil_to_tensor(loader.load_image(seq_id, frame_id2)).to(TENSOR_DEVICE).unsqueeze(0).unsqueeze(0)

        out_decoder, _ = depthnet(img1) # tuple[(b, t, 1, h, w)]

        # flatten batch and seq dim
        out_decoder = out_decoder[0]
        shapes = out_decoder.shape
        out_disp = out_decoder.reshape((-1,) + shapes[2:]) # (b*t, c, h, w)
        out_disp, out_depth = disp_to_depth(out_disp, DEPTH_MIN, DEPTH_MAX)

        out_depth_min = out_depth.min()
        out_depth_max = out_depth.max()
        print("depth max {}, min {}".format(out_depth_max, out_depth_min))
        out_disp_min = out_disp.min()
        out_disp_max = out_disp.max()
        out_disp = (out_disp - out_disp_min)/(out_disp_max - out_disp_min)
        imageio.imwrite(os.path.join(path_result_dir, "img1_disp.png"), tensor_to_pil(out_disp.squeeze(0).cpu()))
        imageio.imwrite(os.path.join(path_result_dir, "img1_ori.png"), tensor_to_pil(img1.squeeze(0).squeeze(0).cpu()))

    else:
        
        fullnet.resume_models(version=version)
        fullnet.eval()
        fullnet.to(TENSOR_DEVICE)

        frame_ids = loader.frames_in_sequence(seq_id)
        frame_id1 = frame_ids[frame_idx - 1]
        frame_id2 = frame_ids[frame_idx]
        img0 = pil_to_tensor(loader.load_image(seq_id, frame_id1)).to(TENSOR_DEVICE).unsqueeze(0)
        img1 = pil_to_tensor(loader.load_image(seq_id, frame_id2)).to(TENSOR_DEVICE).unsqueeze(0)
        imgs = torch.cat((img0, img1), dim=0).unsqueeze(0).to(torch.float)
        K = torch.tensor(loader.load_intrinsics(seq_id)).to(torch.float32).to(TENSOR_DEVICE).unsqueeze(0)

        out_decoder, out_pose_decoder, _ = fullnet(imgs) # (tuple[(b, t, 1, h, w)], Tuple[torch.Tensor, torch.Tensor])
        out_axisangle, out_translation = out_pose_decoder

        # flatten batch and seq dim
        out_decoder = out_decoder[0]
        out_decoder = out_decoder[:,-1:]
        shapes = out_decoder.shape
        out_axisangle = out_axisangle.reshape((-1,) + out_axisangle.shape[2:]) # (b*t, 3)
        out_translation = out_translation.reshape((-1,) + out_translation.shape[2:]) # (b*t, 3)
        out_disp = out_decoder.reshape((-1,) + shapes[2:]) # (b*t, c, h, w)

        # convert axsiangle and translation to matrix
        out_T = Rt_from_axisangle_and_trans(out_axisangle, out_translation) # (b*t, 4, 4)

        out_disp, out_depth = disp_to_depth(out_disp, DEPTH_MIN, DEPTH_MAX)

        # sample
        coors = reproject(out_depth, out_T, K)
        img1_resample = sample(img0, coors)

        # write result
        if not os.path.exists(path_result_dir):
            os.makedirs(path_result_dir)
        
        l1 = (img1 - img1_resample).abs().mean()
        l1_resmaple_before = (img1 - img0).abs().mean()
        print("l1 {} to {}".format(l1_resmaple_before, l1))
        img1_resample_pil = tensor_to_pil(img1_resample.squeeze(0).cpu())
        print("coors max {}, min {}".format(coors.max(), coors.min()))
        print("img1 original max {}, min {}".format(img1.max(), img1.min()))
        print("img1 resample max {}, min {}".format(img1_resample.max(), img1_resample.min()))
        imageio.imwrite(os.path.join(path_result_dir, "img1_resample.png"), img1_resample_pil)

        out_depth_min = out_depth.min()
        out_depth_max = out_depth.max()
        print("depth max {}, min {}".format(out_depth_max, out_depth_min))
        out_disp_min = out_disp.min()
        out_disp_max = out_disp.max()
        out_disp = (out_disp - out_disp_min)/(out_disp_max - out_disp_min)
        img1_depth_pil = tensor_to_pil(out_disp.squeeze(0).cpu())
        imageio.imwrite(os.path.join(path_result_dir, "img1_disp.png"), img1_depth_pil)
        imageio.imwrite(os.path.join(path_result_dir, "img1_ori.png"), tensor_to_pil(img1.squeeze(0).cpu()))
        imageio.imwrite(os.path.join(path_result_dir, "img0_ori.png"), tensor_to_pil(img0.squeeze(0).cpu()))
        print("translation {}".format(out_translation))
        print("axisangle {}".format(out_axisangle))

        print("\nPlease see the generated images in folder \"{}\".".format(path_result_dir))

    pass

if __name__=="__main__":
    main()
