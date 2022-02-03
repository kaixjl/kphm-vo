from itertools import accumulate
import shutil
import subprocess
from typing import Any, Iterable, List, Tuple
from functools import reduce
import os
import os.path
from subprocess import Popen
import pickle
import time
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import argparse
import imageio
from datasets import KittiOdometryDataset, KittiOdometryLoader
from models.misc import create_networks
from utils.torchutils import Rt_from_axisangle_and_trans, reverse_Rt_matrix
from utils.metric import compute_ate
from params import DatasetNameEnum, get_param

param = get_param()

def parse_arg():
    global param

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",   type=str, help="where the dataset is stored")
    parser.add_argument("--config_name", "-c",   type=str, help="specify the tag of the running to be test")
    parser.add_argument("--seq_id", type=str, help="seq id to be test")
    parser.add_argument("--not_cuda", help="use cpu but cuda.", action='store_true')
    parser.add_argument("--archive",   type=str, help="specify archive tag if running archive")
    parser.add_argument("--root",   type=str, help="specify root dir")
    parser.add_argument("--not_generate_pose", help="do not generate pose. use existing pose file.", action='store_true')
    parser.add_argument("--not_compute_ate", help="do not compute ate.", action='store_true')
    parser.add_argument("--use_config", "-u", nargs="+", type=str, help="specify which config to be used.")
    parser.add_argument("--cudas", type=str, help="specify cudas to be used. seperated with comma.")
    parser.add_argument("--test_version", type=int, help="specify version to be test")
    args0 = parser.parse_args()

    param = get_param(args0.use_config)

    def choose(a, b):
        return a if a is not None else b

    param.dataset_dir = choose(args0.dataset_dir, param.dataset_dir)
    param.config_name = choose(args0.config_name, param.config_name)
    param.not_cuda = args0.not_cuda
    param.cuda_visible_devices = choose(args0.cudas, param.cuda_visible_devices)
    param.test_version = choose(args0.test_version, param.test_version)

    if param.dataset_name == DatasetNameEnum.KITTI_ODOM:
        if args0.seq_id is not None:
            args0.seq_id = [int(i) for i in args0.seq_id.split(',')]

    return args0

def main():
    args0 = parse_arg()
    test_pose(param.dataset_dir, param.config_name, args0.archive, args0.root, seq_id=args0.seq_id, not_generate_pose=args0.not_generate_pose, not_compute_ate=args0.not_compute_ate, version=param.test_version)

def test_pose(dataset_dir, config_name, archive=None, root=None, seq_id=None, not_generate_pose=False, not_compute_ate=False, version=0):
    # type: (str, str, str, str, str, bool, bool, int) -> None
    '''
    frame_id and frame_id + 1 will be used.
    '''
    # PARAMETERS
    CONFIG_NAME = config_name
    CUDA = not param.not_cuda
    CUDA_VISIBLE_DEVICES = param.cuda_visible_devices # int or str formatting comma-seperated-integer like "1,2,3,0" is acceptable
    ANGLE_NORMALIZE_FACTOR = None
    NETWORK = param.network
    IMG_HEIGHT = param.img_height
    IMG_WIDTH = param.img_width
    SEQ_IDS = param.test_sequence if seq_id is None else seq_id

    print("SEQ_IDS: {}".format(SEQ_IDS))
    
    # paths
    path_records_dir = os.path.join("records", CONFIG_NAME)
    if archive is not None:
        path_records_dir = os.path.join("archive", archive, path_records_dir)
    if root is not None:
        path_records_dir = os.path.join(root, path_records_dir)
    path_result_dir = os.path.join("result", CONFIG_NAME)

    if os.path.exists(os.path.join(path_records_dir, "hyperparams.pickle")):
        with open(os.path.join(path_records_dir, "hyperparams.pickle"), 'rb') as f:
            hparams = pickle.load(f)
            NETWORK = hparams["NETWORK"]
            IMG_HEIGHT = hparams["IMG_HEIGHT"]
            IMG_WIDTH = hparams["IMG_WIDTH"]
    
    if CUDA_VISIBLE_DEVICES is not None: # set CUDA_VISIBLE_DEVICES
        if type(CUDA_VISIBLE_DEVICES) is int:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_VISIBLE_DEVICES)
        elif type(CUDA_VISIBLE_DEVICES) is tuple or type(CUDA_VISIBLE_DEVICES) is list:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in CUDA_VISIBLE_DEVICES)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    TENSOR_DEVICE = torch.device("cuda") if CUDA and torch.cuda.is_available() else torch.device("cpu")


    # generate pose rel

    if not not_generate_pose:
        # create Network
        print("network")
        fullnet = create_networks(NETWORK, param, False, IMG_HEIGHT, IMG_WIDTH, False)
        if isinstance(fullnet, Iterable):
            fullnet = fullnet[-1]

        print("resume")
        fullnet.set_records_path(path_records_dir)
        fullnet.resume_models(version=version)

        fullnet.eval()
        fullnet.to(TENSOR_DEVICE)
        
        for seq_id in SEQ_IDS:
            if NETWORK == 0:
                fullnet.depth_encoder.reset_convlstm_hidden_state()
                fullnet.pose_encoder.reset_convlstm_hidden_state()
            elif NETWORK == 2:
                fullnet.pose_encoder.reset_convlstm_hidden_state()

            pose_estimated = estimate_seq_pose(seq_id, fullnet, dataset_dir, IMG_HEIGHT, IMG_WIDTH, TENSOR_DEVICE, ANGLE_NORMALIZE_FACTOR)
        
            # write result
            if not os.path.exists(path_result_dir):
                os.makedirs(path_result_dir)
            if isinstance(seq_id, int):
                path_pose = os.path.join(path_result_dir, "{:02d}.txt".format(seq_id))
            else:
                path_pose = os.path.join(path_result_dir, "{}.txt".format(seq_id))
            save_pose(pose_estimated, path_pose)

        # } for seq_id in SEQ_IDS:
    # } if not not_generate_pose:

    # compute ates
    if not not_compute_ate:
        eval_pose2(path_result_dir, SEQ_IDS)
    # } if not not_compute_ate:

    pass
# }

def estimate_seq_pose(seq_id, fullnet, dataset_dir, img_height=128, img_width=416, device=None, ANGLE_NORMALIZE_FACTOR=None):
    '''
    estimated pose through a kitti sequence

    ## Return

    (b, 4, 4) poses (include the 1st frame with identity matrix)
    '''
    # create Dataset and DataLoader object
    print("test pose of seq_id={}".format(seq_id))
    dataset = KittiOdometryDataset(seq_id, dataset_dir=dataset_dir, img_height=img_height, img_width=img_width, return_intrinsics=True)
    dl = DataLoader(dataset, 1)

    # pose_rel_estimated = []
    pose_estimated = []

    start_time = time.time()

    last_frame = None # type: torch.Tensor
    for i, item in enumerate(dl):
        print("\r{:4d}".format(i), flush=True, end="")
        frame, _ = item

        if last_frame is None:
            eye = torch.eye(4, device=device)
            # pose_rel_estimated.append(eye)
            pose_estimated.append(eye)
            last_frame = frame # type: torch.Tensor
            continue

        # Iterative
        img0 = last_frame
        img1 = frame
        imgs = torch.cat((img0, img1), dim=0).unsqueeze(0).to(torch.float).to(device)

        _, out_pose_decoder, _ = fullnet(imgs) # (tuple[(b, t, 1, h, w)], Tuple[torch.Tensor, torch.Tensor])
        out_axisangle, out_translation = out_pose_decoder

        # flatten batch and seq dim
        out_axisangle = out_axisangle.reshape((-1,) + out_axisangle.shape[2:]) # (b*t, 3)
        out_translation = out_translation.reshape((-1,) + out_translation.shape[2:]) # (b*t, 3)

        # convert axsiangle and translation to matrix
        out_T = Rt_from_axisangle_and_trans(out_axisangle, out_translation, ANGLE_NORMALIZE_FACTOR) # (b*t, 4, 4)

        # save pose estimated
        out_T = out_T[0].detach()
        # pose_rel_estimated.append(out_T)
        pose_estimated.append(torch.matmul(pose_estimated[-1], out_T))

        # save last frame
        last_frame = frame
    print("\r{}\nseq_id {} finished.".format(len(dl), seq_id))
    
    total_time = time.time() - start_time
    print("totally cost {} s. {} s per frame. {} fps.".format(total_time, total_time/len(dl), len(dl)/total_time))
    
    return torch.stack(pose_estimated)

def save_pose(poses, path_file):
    # type: (torch.Tensor, str) -> None
    '''
    save poses as a file. Each line is the pose of a frame as 12 elements of the corresponding matrix.

    ## Parameters

    - poses: (b, 4, 4)
    - path_file: where to save
    '''
    poses = poses[:, :3, :].reshape((-1, 12)).detach().cpu().numpy() # type: np.ndarray
    np.savetxt(path_file, poses)

def load_pose(path_file):
    # type: (str) -> torch.Tensor
    '''
    load pose file saved by save_pose

    ## Return:

    (b, 4, 4)
    '''
    poses = np.loadtxt(path_file, dtype=np.float32).reshape((-1, 3, 4))
    last_line = np.zeros((poses.shape[0], 1, 4), dtype=np.float32)
    last_line[:,0,3] = 1
    poses = np.concatenate((poses, last_line), axis=1)
    assert(poses.shape[1]==4 and poses.shape[2]==4)
    return torch.from_numpy(poses)

def eval_pose2(path_result_dir, seqs):
    
    from kitti_eval.kitti_odometry import KittiEvalOdom

    eval_tool = KittiEvalOdom()
    gt_dir = "./kitti_eval/gt_poses/"
    result_dir = path_result_dir

    print("evaluating pose...")

    eval_tool.eval(
        gt_dir,
        result_dir,
        alignment="7dof",
        seqs=seqs,
    )
    
    print("evaluation finished.")

if __name__=="__main__":
    main()
