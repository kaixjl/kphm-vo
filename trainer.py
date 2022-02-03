from typing import Callable, Dict, Tuple, Iterable, List, Union
from functools import reduce
import os
import os.path
import sys
import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter
from datasets.odometry_dataset import OdometryDatasetGTEnum
from datasets import *
from models.fullnet import FullnetBase
from models.misc import create_networks
from utils.torchutils import seq_adjacent_concat, sample, reproject, Rt_from_axisangle_and_trans, disp_to_depth, reverse_Rt_matrix, Reproject, pose_abs_to_rel
from utils.loss import SSIMLoss, disp_smooth_loss
from utils import Timer
from params import DatasetNameEnum, Parameters

notify_can_not_with_dispreprojection = False

class ParamAdjust():
    def __init__(self, t):
        # type: (Train, int) -> None
        self.t = t
        self._epoch = 0
        self._step = 0
        self._total_epoch = 0

    def step(self):
        self._step += 1
        self.new_step(self._step)

    def step_epoch(self):
        self._epoch += 1
        self.new_epoch(self._epoch, self._total_epoch)

    def step_train(self, total_epoch):
        self._epoch = 0
        self._total_epoch = total_epoch
        self.new_train(self._total_epoch)

    def on_start(self):
        self.new_start()
    
    @property
    def Trainer(self):
        return self.t

    def __call__(self):
        self.step()

    def new_epoch(self, epoch, total_epoch):
        pass

    def new_step(self, step):
        pass

    def new_train(self, total_epoch):
        pass

    def new_start(self):
        pass

class TrainParamAdjust(ParamAdjust):
    def __init__(self, t):
        super().__init__(t)

    def new_epoch(self, epoch, total_epoch):
        return super().new_epoch(epoch, total_epoch)
        # if epoch == total_epoch // 2:
        #     self.Trainer.fullnet.TRAIN_POSE_ENCODER=False

    def new_step(self, step):
        return super().new_step(step)

    def new_train(self, total_epoch):
        pass

    def new_start(self):
        pass

class Train():
    def __init__(self, param, create_network=False):
        # type: (Parameters, bool) -> None
        self.param = param
        self.train_inited = False
        self.train_started = False

        self.init(create_network)

    def init(self, create_network=False):
        param = self.param
        # PARAMETERS
        self.CONFIG_NAME = param.config_name
        self.CUDA = not param.not_cuda
        self.IMG_HEIGHT = param.img_height
        self.IMG_WIDTH = param.img_width
        self.DEPTH_MAX = param.depth_max
        self.DEPTH_MIN = param.depth_min
        self.CUDA_VISIBLE_DEVICES = param.cuda_visible_devices # int or str formatting comma-seperated-integer like "1,2,3,0" is acceptable
        self.NETWORK = param.network # refer to create_networks
        # self.SHARE_ENCODER = param.share_encoder
        self.LOAD_PRETRAINED = param.pretrained
        # self.CONCAT_TYPE = 0 # 0 for seq_adjacent_concat, 1 for seq_first_seq

        if self.CUDA_VISIBLE_DEVICES is not None: # set CUDA_VISIBLE_DEVICES
            if type(self.CUDA_VISIBLE_DEVICES) is int:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.CUDA_VISIBLE_DEVICES)
            elif type(self.CUDA_VISIBLE_DEVICES) is tuple or type(self.CUDA_VISIBLE_DEVICES) is list:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.CUDA_VISIBLE_DEVICES)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.CUDA_VISIBLE_DEVICES

        self.TENSOR_DEVICE = torch.device("cuda") if self.CUDA and torch.cuda.is_available() else torch.device("cpu")
        # self.SEQ_CONCAT = CONCAT_TYPE_DICT[self.CONCAT_TYPE]

        # paths
        self.path_records_dir = os.path.join("records", self.CONFIG_NAME)
        self.path_result_dir = os.path.join("result", self.CONFIG_NAME)

        self.fullnet = None
        if create_network:
            fullnet = create_networks(self.NETWORK, param, self.LOAD_PRETRAINED, self.IMG_HEIGHT, self.IMG_WIDTH, True)
            self.set_networks(fullnet)

    def train_init(self):
        if self.fullnet is None:
            raise Exception("Train.set_networks should be invoked before Train.train_init is called.")

        param = self.param
        # Training Parameters
        self.EPOCH_SIZE = param.epoch_size
        self.RESUME = param.resume
        self.LR = param.lr
        self.LR_DEPTH_ENCODER = self.LR * 1.e-0
        self.LR_DEPTH_DECODER = self.LR * 1.e-0
        self.LR_POSE_ENCODER = self.LR * 1.e-0
        self.LR_POSE_DECODER = self.LR * 1.e-0
        self.WEIGHT_DECAY = param.weight_decay
        self.WEIGHT_DECAY_DEPTH_ENCODER = self.WEIGHT_DECAY
        self.WEIGHT_DECAY_DEPTH_DECODER = self.WEIGHT_DECAY
        self.WEIGHT_DECAY_POSE_ENCODER = self.WEIGHT_DECAY
        self.WEIGHT_DECAY_POSE_DECODER = self.WEIGHT_DECAY
        self.BETAS_ADAM = param.betas_adam
        self.BATCH_SIZE = param.batch_size
        self.SSIM_LOSS_FACTOR = param.ssim_loss_factor
        self.L1_LOSS_FACTOR = param.l1_loss_factor
        self.L2_LOSS_FACTOR = param.l2_loss_factor
        self.DISP_SMOOTH_LOSS_FACTOR = param.disp_smooth_loss_factor
        self.SPLIT_TAG = param.split_tag
        self.WITH_AUTOMASK = param.WITH_AUTOMASK
        self.WITH_KPHEATMAP_WEIGHTING = param.WITH_KPHEATMAP_WEIGHTING
        self.WITH_KPHEATMAP_MASK = param.WITH_KPHEATMAP_MASK
        self.WITH_KPHEATMAP = param.WITH_KPHEATMAP
        self.LR_SCHEDULER_GAMMA = param.lr_scheduler_gamma
        self.HEATMAP_RADIUS = param.heatmap_radius
        self.HEATMAP_KERNEL_FUNC = param.heatmap_kernel_func
        self.KEYPOINT = param.keypoint
        self.WITH_ACC_POSE = param.WITH_ACC_POSE
        self.WITH_AUG = param.WITH_AUG
        self.COMMENT = param.comment
        self.VALID_SEQ_IDS = param.valid_sequence
        self.DATASET_NAME = param.dataset_name
        self.IS_DISTRIBUTED = param.is_distributed > 0 and dist.is_available()
        self.RANK = dist.get_rank() if self.IS_DISTRIBUTED else None

        self.TENSOR_DEVICE = torch.device("cuda:{}".format(self.RANK)) if self.CUDA and self.IS_DISTRIBUTED and torch.cuda.is_available() else self.TENSOR_DEVICE

        self.path_split_note = "splits/note.txt" if self.SPLIT_TAG is None else os.path.join("splits", self.SPLIT_TAG, "note.txt")
        if os.path.exists(self.path_split_note):
            with open(self.path_split_note, 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    line = line.split(' ')
                    if line[0]=="val":
                        self.VALID_SEQ_IDS = tuple(i for i in line[1:])
                        break

        self.hyperparams_dict = {"COMMENT": self.COMMENT,
                            "DATASET_NAME": self.DATASET_NAME,
                            "EPOCH_SIZE": self.EPOCH_SIZE,
                            "CONFIG_NAME": self.CONFIG_NAME,
                            "CUDA": self.CUDA,
                            "IS_DISTRIBUTED": self.IS_DISTRIBUTED,
                            "RANK": self.RANK,
                            "RESUME": self.RESUME,
                            "LOAD_PRETRAINED": self.LOAD_PRETRAINED,
                            "LR": self.LR,
                            "LR_DEPTH_ENCODER": self.LR_DEPTH_ENCODER,
                            "LR_DEPTH_DECODER": self.LR_DEPTH_DECODER,
                            "LR_POSE_ENCODER": self.LR_POSE_ENCODER,
                            "LR_POSE_DECODER": self.LR_POSE_DECODER,
                            "WEIGHT_DECAY": self.WEIGHT_DECAY,
                            "WEIGHT_DECAY_DEPTH_ENCODER": self.WEIGHT_DECAY_DEPTH_ENCODER,
                            "WEIGHT_DECAY_DEPTH_DECODER": self.WEIGHT_DECAY_DEPTH_DECODER,
                            "WEIGHT_DECAY_POSE_ENCODER": self.WEIGHT_DECAY_POSE_ENCODER,
                            "WEIGHT_DECAY_POSE_DECODER": self.WEIGHT_DECAY_POSE_DECODER,
                            "BETAS_ADAM": self.BETAS_ADAM,
                            "IMG_HEIGHT": self.IMG_HEIGHT,
                            "IMG_WIDTH": self.IMG_WIDTH,
                            "BATCH_SIZE": self.BATCH_SIZE,
                            "CUDA_VISIBLE_DEVICES": self.CUDA_VISIBLE_DEVICES,
                            "SSIM_LOSS_FACTOR": self.SSIM_LOSS_FACTOR,
                            "L1_LOSS_FACTOR": self.L1_LOSS_FACTOR,
                            "L2_LOSS_FACTOR": self.L2_LOSS_FACTOR,
                            "DISP_SMOOTH_LOSS_FACTOR": self.DISP_SMOOTH_LOSS_FACTOR,
                            "NETWORK": self.NETWORK,
                            "SPLIT_TAG": self.SPLIT_TAG,
                            "LR_SCHEDULER_GAMMA": self.LR_SCHEDULER_GAMMA,
                            "WITH_AUTOMASK": self.WITH_AUTOMASK,
                            "WITH_KPHEATMAP": self.WITH_KPHEATMAP,
                            "WITH_KPHEATMAP_WEIGHTING": self.WITH_KPHEATMAP_WEIGHTING,
                            "WITH_KPHEATMAP_MASK": self.WITH_KPHEATMAP_MASK,
                            "HEATMAP_RADIUS": self.HEATMAP_RADIUS,
                            "HEATMAP_KERNEL_FUNC": self.HEATMAP_KERNEL_FUNC,
                            "KEYPOINT": self.KEYPOINT,
                            "WITH_ACC_POSE": self.WITH_ACC_POSE,
                            "WITH_AUG": self.WITH_AUG,
                            "VALID_SEQ_IDS": self.VALID_SEQ_IDS,
                            }

        self.path_dataset_related_description = os.path.join(param.dataset_related_dir, "description.txt")
        if os.path.exists(self.path_dataset_related_description):
            with open(self.path_dataset_related_description, 'r') as f:
                desc = f.read().splitlines()
            desc = OrderedDict(list(map(lambda x: x.strip(), d.split(":"))) for d in desc)
            self.hyperparams_dict.update(desc)

        # paths
        self.path_sw_dir = os.path.join("runs", self.CONFIG_NAME)

        self.path_records_hyperparameters = os.path.join(self.path_records_dir, "hyperparams.pickle")
        self.path_result_hyperparameters_txt = os.path.join(self.path_result_dir, "hyperparams.txt")

        self.ssim_loss = SSIMLoss()
        # self.l1_loss = L1Loss()
        # self.l2_loss = MSELoss()
        self.l1_loss = lambda x, y: torch.abs(x - y) # type: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        self.l2_loss = lambda x, y: (x - y) ** 2 # type: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

        # BackprojectDepth and Project3D Object
        self.reproject = Reproject(self.IMG_HEIGHT, self.IMG_WIDTH, device=self.TENSOR_DEVICE)


        # initiate variables
        self.global_step = 0
        self.epoch_save_models = -1
        
        print("{\n", "\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), self.hyperparams_dict.items())), "\n}", sep="")

        # create Dataset and DataLoader object
        print("dataset and dataloader")
        dls = Train.create_dateloader(self.DATASET_NAME, param.dataset_dir, param.dataset_related_dir, self.BATCH_SIZE, self.SPLIT_TAG, self.IMG_HEIGHT, self.IMG_WIDTH, self.WITH_KPHEATMAP, self.WITH_AUG, self.IS_DISTRIBUTED)
        self.set_dataloader(*dls)

        self.train_inited = True
    
    def set_networks(self, fullnet):
        # type: (FullnetBase) -> None

        self.fullnet = fullnet
        self.fullnet.set_records_path(self.path_records_dir)
        self.train_inited = False

    @staticmethod
    def create_dateloader(dataset_name, dataset_dir, dataset_related_dir, batch_size, split_tag, img_height, img_width, kphm=False, aug=False, code_input=False, is_distributed=False, aug_cut=False):
        dl, dl_val, sampler, sampler_val = None, None, None, None
        path_train_split = "splits/train.txt" if split_tag is None else os.path.join("splits", split_tag, "train.txt")
        path_val_split = "splits/val.txt" if split_tag is None else os.path.join("splits", split_tag, "val.txt")
        if dataset_name == DatasetNameEnum.KITTI_ODOM:
            dataset = KittiOdometrySequenceWithHeatmapDataset(path_train_split, dataset_dir=dataset_dir, dataset_related_dir=dataset_related_dir, img_height=img_height, img_width=img_width, return_intrinsics=True, gt_type=OdometryDatasetGTEnum.NONE, return_heatmap=kphm, aug=aug)
            sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
            dl = DataLoader(dataset, batch_size, shuffle=(sampler is None), drop_last=True, sampler=sampler)
            
            dataset_val = KittiOdometrySequenceWithHeatmapDataset(path_val_split, dataset_dir=dataset_dir, dataset_related_dir=dataset_related_dir, img_height=img_height, img_width=img_width, return_intrinsics=True, gt_type=OdometryDatasetGTEnum.NONE, return_heatmap=kphm)
            sampler_val = DistributedSampler(dataset_val, shuffle=True) if is_distributed else None
            dl_val = DataLoader(dataset_val, batch_size, shuffle=(sampler_val is None), drop_last=True, sampler=sampler_val)

        return dl, dl_val, sampler, sampler_val

    def set_dataloader(self, dl, dl_val=None, sampler=None, sampler_val=None):
        self.dl, self.dl_val, self.sampler, self.sampler_val = dl, dl_val, sampler, sampler_val

    def save_models(self):
        if not self.IS_DISTRIBUTED or self.RANK==0:

            if not os.path.exists(self.path_records_dir):
                os.makedirs(self.path_records_dir)
            self.fullnet.save_models()

        if self.IS_DISTRIBUTED: dist.barrier()

    def save_training_models(self):
        if not self.IS_DISTRIBUTED or self.RANK==0:

            if not os.path.exists(self.path_records_dir):
                os.makedirs(self.path_records_dir)
            if not os.path.exists(self.path_result_dir):
                os.makedirs(self.path_result_dir)

            self.fullnet.save_optimizers()

            self.fullnet.save_lr_schedulers()
            
            self.hyperparams_dict["global_step"] = self.global_step
            self.hyperparams_dict["loss_val_min"] = self.loss_val_min
            with open(self.path_records_hyperparameters, 'wb') as f:
                pickle.dump(self.hyperparams_dict, f)
            with open(self.path_result_hyperparameters_txt, 'w') as f:
                f.write("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), self.hyperparams_dict.items())))


        if self.IS_DISTRIBUTED: dist.barrier()
        pass
    # }

    def resume_models(self, device=None):
        self.fullnet.resume_models(device=device)

        if self.IS_DISTRIBUTED: dist.barrier()

    def resume_training_models(self, device=None):
        self.fullnet.resume_optimizers(device=device)
        self.fullnet.resume_lr_schedulers(device=device)
        with open(self.path_records_hyperparameters, 'rb') as f:
            hparams = pickle.load(f)
            self.global_step = hparams["global_step"] if "global_step" in hparams.keys() else self.global_step
            self.loss_val_min = hparams["loss_val_min"] if "loss_val_min" in hparams.keys() else self.loss_val_min

        if self.IS_DISTRIBUTED: dist.barrier()
        pass
    # }

    def print_losses(self,
                     loss_dict, # type: Dict[str, torch.Tensor]
                     ):
        if not self.IS_DISTRIBUTED or self.RANK==0:

            print("\x1b[u", end="") ## VT100 
            for k, v in loss_dict.items():
                print("{}: {}".format(k, v.cpu()))
            print("===", flush=True)

        if self.IS_DISTRIBUTED: dist.barrier()

    def record_losses(self,
                      loss_dict, # type: Dict[str, torch.Tensor]
                      step):
        if not self.IS_DISTRIBUTED or self.RANK==0:

            for k, v in loss_dict.items():
                self.sw.add_scalar("loss/{}".format(k), v.cpu().item(), step)

        if self.IS_DISTRIBUTED: dist.barrier()

    def compute_loss(self, img_seq, out_disps):
        '''
        img_seq: (b, T, c, h, w)
        out_disps: [(b, T, c, h, w)]
        '''
        shapes = img_seq.shape
        img_seq_flat = img_seq.flatten(0, 1) # (b*t, c, h, w)
        out_disps_flat = [i.flatten(0, 1) for i in out_disps] # [(b*t, c, h, w)]

        rst_disp_smooth_loss = torch.stack([disp_smooth_loss(i, img_seq_flat) for i in out_disps_flat])

        rst_disp_smooth_loss = rst_disp_smooth_loss.reshape((-1,) + shapes[:2] + rst_disp_smooth_loss.shape[2:])

        return rst_disp_smooth_loss
    # } def compute_loss

    def compute_reprojection_loss(self, img_seq_former, img_seq_latter, K, out_T, out_depths_latter, gt, heatmap=None,
            WITH_AUTOMASK=False):
        '''
        ## Parameter:

        - img_seq_former, img_seq_latter: (b, T-1, c, h, w)
        - K: (b, 3, 3)
        - out_T: (b, T-1, 4, 4), latter -> former
        - out_depths_latter: # [(b, T-1, c, h, w)]
        - gt: (b, T-1, 4, 4)
        - heatmap: (b, T, c, h, w)
        '''
        shapes = img_seq_former.shape
        img_seq_former_flat = img_seq_former.flatten(0, 1) # type: torch.Tensor # (b*t, c, h, w)
        img_seq_latter_flat = img_seq_latter.flatten(0, 1) # type: torch.Tensor # (b*t, c, h, w)
        K_flat = K.unsqueeze(1).expand(shapes[:2] + K.shape[1:]).flatten(0, 1) # (b, 3, 3) -> (b, t, 3, 3) -> (b*t, 3, 3)
        out_T_flat = out_T.reshape((-1,) + out_T.shape[2:]) # (b*t, 4, 4)
        out_depths_latter_flat = [i.flatten(0, 1) for i in out_depths_latter] # [(b*t, c, h, w)]

        # sample
        coorss = [reproject(i, out_T_flat, K_flat) for i in out_depths_latter_flat]
        img_seq_latter_resamples = [sample(img_seq_former_flat, i) for i in coorss]

        # compute loss
        rst_ssim_loss = torch.stack([self.ssim_loss(i, img_seq_latter_flat).mean(dim=1, keepdim=True) for i in img_seq_latter_resamples]) # (n_depth_scales, b*t, c, h, w)
        rst_l1_loss = torch.stack([self.l1_loss(i, img_seq_latter_flat).mean(dim=1, keepdim=True) for i in img_seq_latter_resamples]) # (n_depth_scales, b*t, c, h, w)
        rst_l2_loss = torch.stack([self.l2_loss(i, img_seq_latter_flat).mean(dim=1, keepdim=True) for i in img_seq_latter_resamples]) # (n_depth_scales, b*t, c, h, w)
        rst_ssim_loss_raw = rst_ssim_loss.detach()
        rst_l1_loss_raw = rst_l1_loss.detach()
        rst_l2_loss_raw = rst_l2_loss.detach()

        mask = torch.ones(rst_ssim_loss.shape[:1] + shapes[:2] + (1,) + rst_ssim_loss.shape[3:]).to(self.TENSOR_DEVICE).to(torch.bool) # (n_depth_scales, b, t, 1, h, w)

        if WITH_AUTOMASK:
            with torch.no_grad():
                rst_ori_ssim = self.ssim_loss(img_seq_former_flat, img_seq_latter_flat).mean(dim=1, keepdim=True).unsqueeze(0).expand_as(rst_ssim_loss) # (n_depth_scales, b*t, c, h, w)
                rst_ori_l1 = self.l1_loss(img_seq_former_flat, img_seq_latter_flat).mean(dim=1, keepdim=True).unsqueeze(0).expand_as(rst_l1_loss) # (n_depth_scales, b*t, c, h, w)
                rst_ori_l2 = self.l2_loss(img_seq_former_flat, img_seq_latter_flat).mean(dim=1, keepdim=True).unsqueeze(0).expand_as(rst_l2_loss) # (n_depth_scales, b*t, c, h, w)

                rst_ssim_loss_weight = (torch.cat((rst_ssim_loss, rst_ori_ssim), dim=2).argmin(dim=2, keepdim=True)==0).to(torch.float32)
                rst_l1_loss_weight = (torch.cat((rst_l1_loss, rst_ori_l1), dim=2).argmin(dim=2, keepdim=True)==0).to(torch.float32)
                rst_l2_loss_weight = (torch.cat((rst_l2_loss, rst_ori_l2), dim=2).argmin(dim=2, keepdim=True)==0).to(torch.float32)

            rst_ssim_loss *= rst_ssim_loss_weight
            rst_l1_loss *= rst_l1_loss_weight
            rst_l2_loss *= rst_l2_loss_weight

        # unflatten
        rst_ssim_loss = rst_ssim_loss.reshape((-1,) + shapes[:2] + rst_ssim_loss.shape[2:])
        rst_l1_loss = rst_l1_loss.reshape((-1,) + shapes[:2] + rst_l1_loss.shape[2:])
        rst_l2_loss = rst_l2_loss.reshape((-1,) + shapes[:2] + rst_l2_loss.shape[2:])

        if heatmap is not None:
            heatmap = heatmap.unsqueeze(0)[:,:,1:] # (n_depth_scales, b, T-1, c, h, w)
            if self.WITH_KPHEATMAP_WEIGHTING:
                rst_ssim_loss = rst_ssim_loss * heatmap
                rst_l1_loss = rst_l1_loss * heatmap
                rst_l2_loss = rst_l2_loss * heatmap

            if self.WITH_KPHEATMAP_MASK:
                mask = mask and heatmap>0

        return ComputeReprojectionLossResult(rst_ssim_loss, rst_l1_loss, rst_l2_loss, mask, rst_ssim_loss_raw, rst_l1_loss_raw, rst_l2_loss_raw)
    # } def compute_reprojection_loss

    def compute_reprojection_loss_acc(self, img_seq_former, img_seq_latter, K, out_T, out_depths_latter, gt, t_cut_lim, heatmap=None,
            WITH_AUTOMASK=False):
        img_seq_former_cut = img_seq_former
        img_seq_latter_cut = img_seq_latter
        K_cut = K
        out_T_cut = out_T
        gt_cut = gt
        out_depths_latter_cut = out_depths_latter
        heatmap_cut = heatmap
        rst_ssim_loss, rst_l1_loss, rst_l2_loss, mask, rst_ssim_loss_raw, rst_l1_loss_raw, rst_l2_loss_raw = self.compute_reprojection_loss(img_seq_former_cut, img_seq_latter_cut, K_cut, out_T_cut, out_depths_latter_cut, gt_cut, heatmap_cut, WITH_AUTOMASK).deconstruct()

        rst_ssim_loss_list = [rst_ssim_loss]
        rst_l1_loss_list = [rst_l1_loss]
        rst_l2_loss_list = [rst_l2_loss]
        rst_ssim_loss_raw_list = [rst_ssim_loss_raw]
        rst_l1_loss_raw_list = [rst_l1_loss_raw]
        rst_l2_loss_raw_list = [rst_l2_loss_raw]
        mask_list = [mask]

        for t_cut in range(1, t_cut_lim):
            img_seq_former_cut = img_seq_former_cut[:, :-1]
            img_seq_latter_cut = img_seq_latter_cut[:, 1:]
            K_cut = K_cut
            out_T_cut = torch.matmul(out_T_cut[:,:-1], out_T[:,t_cut:])
            gt_cut = torch.matmul(gt_cut[:,:-1], gt[:,t_cut:]) if gt_cut is not None else gt_cut
            out_depths_latter_cut = [i[:, 1:] for i in out_depths_latter_cut]
            heatmap_cut = heatmap_cut[:, 1:] if heatmap is not None else None
            rst_ssim_loss, rst_l1_loss, rst_l2_loss, mask, rst_ssim_loss_raw, rst_l1_loss_raw, rst_l2_loss_raw = self.compute_reprojection_loss(img_seq_former_cut, img_seq_latter_cut, K_cut, out_T_cut, out_depths_latter_cut, gt_cut, heatmap_cut, WITH_AUTOMASK).deconstruct()

            rst_ssim_loss_list.append(rst_ssim_loss)
            rst_l1_loss_list.append(rst_l1_loss)
            rst_l2_loss_list.append(rst_l2_loss)
            rst_ssim_loss_raw_list = [rst_ssim_loss_raw]
            rst_l1_loss_raw_list = [rst_l1_loss_raw]
            rst_l2_loss_raw_list = [rst_l2_loss_raw]
            mask_list.append(mask)

        return ComputeReprojectionLossAccResult(rst_ssim_loss_list, rst_l1_loss_list, rst_l2_loss_list, mask=mask_list, rst_ssim_loss_raw=rst_ssim_loss_raw_list, rst_l1_loss_raw=rst_l1_loss_raw_list, rst_l2_loss_raw=rst_l2_loss_raw_list)
    # } def compute_reprojection_loss_acc()

    def forward(self, fullnet, img_seq, K=None, gt=None):
        '''
        ## Parameters:

        - img_seq, heatmap: (b, T, c, h, w)
        - K: (b, 3, 3)
        - gt: (b, T-1, 4, 4)
        '''
        out_decoder, out_pose_decoder, ret = fullnet(img_seq) # (tuple[(b, t, 1, h, w)], Tuple[torch.Tensor, torch.Tensor], Dict of ret)
        return out_decoder, out_pose_decoder, ret

    def proc_one_batch_forward(self, fullnet, img_seq, K, gt, heatmap=None, t_cut_lim=None,
            WITH_DISP_SMOOTH=True, WITH_AUTOMASK=False):
        '''
        process on batch forward

        ## Parameters:

        - img_seq, heatmap: (b, T, c, h, w)
        - K: (b, 3, 3)
        - gt: (b, T-1, 4, 4)
        - heatmap: (b, T, 4, 4)

        ## Return:

        ProcOneBatchResult obejct
        '''
        img_seq = img_seq.to(torch.float)
        img_adj = seq_adjacent_concat(img_seq).to(torch.float)

        shapes_seq = img_seq.shape
        shapes_adj = img_adj.shape # save for batch_size and seq_len
        if t_cut_lim is None:
            t_cut_lim = shapes_adj[1]

        # input image into network
        fullnet_input = img_seq
        out_decoder, out_pose_decoder, ret = self.forward(fullnet, fullnet_input, K, gt) # (tuple[(b, t, 1, h, w)], Tuple[torch.Tensor, torch.Tensor])
        out_axisangle, out_translation = out_pose_decoder

        # out_disps of all or the latter
        out_disps_shape = out_decoder[0].shape
        is_out_disps_all = out_disps_shape[1]==shapes_seq[1]

        # flatten in batch and seq dim
        out_axisangle = out_axisangle.reshape((-1,) + out_axisangle.shape[2:]) # (b*t, 3)
        out_translation = out_translation.reshape((-1,) + out_translation.shape[2:]) # (b*t, 3)
        out_disps = [F.interpolate(i.reshape((-1,) + i.shape[2:]), size=(self.IMG_HEIGHT, self.IMG_WIDTH), mode="bilinear") for i in out_decoder] # [(b*t, c, h, w)]
        out_disps_and_depths = [disp_to_depth(i, self.DEPTH_MIN, self.DEPTH_MAX) for i in out_disps] # [(b*t, c, h, w), (b*t, c, h, w)]
        out_depths = [i[1] for i in out_disps_and_depths] # [(b*t, c, h, w)]
        # out_disps = [i[0] for i in out_disps_and_depths] # [(b*t, c, h, w)]

        # convert axsiangle and translation to matrix
        out_T = Rt_from_axisangle_and_trans(out_axisangle, out_translation) # (b*t, 4, 4)
        out_T = out_T.reshape(shapes_adj[:2] + out_T.shape[1:]) # (b, t, 4, 4)
        
        out_disps = [i.reshape(out_disps_shape[:2] + i.shape[1:]) for i in out_disps] # [(b, t, c, h, w)]
        out_depths = [i.reshape(out_disps_shape[:2] + i.shape[1:]) for i in out_depths] # [(b, t, c, h, w)]
        if is_out_disps_all:
            out_depths_latter = [i[:,1:] for i in out_depths] # [(b, t, c, h, w)]
        else:
            out_depths_latter = out_depths
        
        chs_per_img = img_adj.shape[2] // 2
        img_seq_former = img_adj[:,:,:chs_per_img] # (b, t, 3, h, w)
        img_seq_latter = img_adj[:,:,chs_per_img:] # (b, t, 3, h, w)
        rst = self.compute_reprojection_loss_acc(img_seq_former, img_seq_latter, K, out_T, out_depths_latter, gt, t_cut_lim, heatmap=heatmap,
            WITH_AUTOMASK=WITH_AUTOMASK)

        rst_disp_smooth_loss = None
        if WITH_DISP_SMOOTH:
            if is_out_disps_all:
                rst_disp_smooth_loss = self.compute_loss(img_seq, out_disps)
            else:
                rst_disp_smooth_loss = self.compute_loss(img_seq_latter, out_disps)

        return ProcOneBatchForwardResult(rst, rst_disp_smooth_loss)
    # } def proc_one_batch_forward

    def proc_one_batch(self, fullnet, img_seq, K, gt, heatmap=None,
            WITH_ACC_POSE=False, WITH_AUTOMASK=False):

        forward_rst = self.proc_one_batch_forward(fullnet, img_seq, K, gt, heatmap, None if WITH_ACC_POSE else 1,
            WITH_DISP_SMOOTH=True, WITH_AUTOMASK=WITH_AUTOMASK)

        rst_ssim_loss = forward_rst.reprojection_loss.rst_ssim_loss
        rst_l1_loss = forward_rst.reprojection_loss.rst_l1_loss
        rst_l2_loss = forward_rst.reprojection_loss.rst_l2_loss
        rst_disp_smooth_loss = forward_rst.rst_disp_smooth_loss
        mask = forward_rst.reprojection_loss.mask

        rst_ssim_loss = torch.cat([rst_ssim_loss[i][mask[i]] for i in range(len(mask))]).mean()
        rst_l1_loss = torch.cat([rst_l1_loss[i][mask[i]] for i in range(len(mask))]).mean()
        rst_l2_loss = torch.cat([rst_l2_loss[i][mask[i]] for i in range(len(mask))]).mean()
        rst_disp_smooth_loss = rst_disp_smooth_loss.mean()
        
        losses = []
        losses.append(rst_ssim_loss * self.SSIM_LOSS_FACTOR)
        losses.append(rst_l1_loss * self.L1_LOSS_FACTOR)
        losses.append(rst_l2_loss * self.L2_LOSS_FACTOR)
        losses.append(rst_disp_smooth_loss * self.DISP_SMOOTH_LOSS_FACTOR)

        rst_ssim_loss_raw = forward_rst.reprojection_loss.rst_ssim_loss_raw
        rst_l1_loss_raw = forward_rst.reprojection_loss.rst_l1_loss_raw
        rst_l2_loss_raw = forward_rst.reprojection_loss.rst_l2_loss_raw
        rst_ssim_loss_raw = torch.cat([rst_ssim_loss_raw[i][:] for i in range(len(rst_ssim_loss_raw))]).mean()
        rst_l1_loss_raw = torch.cat([rst_l1_loss_raw[i][:] for i in range(len(rst_l1_loss_raw))]).mean()
        rst_l2_loss_raw = torch.cat([rst_l2_loss_raw[i][:] for i in range(len(rst_l2_loss_raw))]).mean()
        losses_raw = []
        losses_raw.append(rst_ssim_loss_raw * self.SSIM_LOSS_FACTOR)
        losses_raw.append(rst_l1_loss_raw * self.L1_LOSS_FACTOR)
        losses_raw.append(rst_l2_loss_raw * self.L2_LOSS_FACTOR)

        loss = reduce(lambda x, y: x + y, losses) # type: torch.Tensor
        loss_raw = reduce(lambda x, y: x + y, losses_raw) # type: torch.Tensor

        # return ProcOneBatchResult(rst_ssim_loss, rst_l1_loss, rst_l2_loss, rst_disp_smooth_loss, rst_coors_loss, rst_gt_reproj_ssim_loss, loss=loss)
        return ProcOneBatchResult(rst_ssim_loss, rst_l1_loss, rst_l2_loss, rst_disp_smooth_loss, loss=loss, loss_raw=loss_raw)
    # } def proc_one_batch

    def eval_valid(self, dl_val):
        # type: (DataLoader) -> None
        fullnet = self.fullnet_ddp if self.IS_DISTRIBUTED else self.fullnet
        fullnet.eval()
        loss_val = []
        with torch.no_grad():
            for eval_step, item in enumerate(dl_val):
                if eval_step % 200 == 0:
                    print("\r{}".format(eval_step), flush=True, end="")
                heatmap = None
                item_idx = 2

                # img_seq, K, gt = item[:item_idx]
                img_seq, K = item[:item_idx]

                if self.WITH_KPHEATMAP:
                    heatmap = item[item_idx]
                    item_idx += 1

                img_seq = img_seq.to(self.TENSOR_DEVICE)
                K = K.to(self.TENSOR_DEVICE)
                gt = None #gt.to(self.TENSOR_DEVICE)
                
                if heatmap is not None:
                    heatmap = heatmap.to(self.TENSOR_DEVICE)

                rst = self.proc_one_batch(fullnet, img_seq, K, gt, WITH_ACC_POSE=self.WITH_ACC_POSE)
                loss_raw = rst.loss_raw
                
                loss_val.append(loss_raw)

                pass
            # } for item in dl_val
            print()
            loss_val = torch.tensor(loss_val).mean()
        # } with torch.no_grad()

        fullnet.train()

        return loss_val.detach().cpu().item()
    # } def eval_valid()

    def train_one_epoch(self, dl):
        fullnet = self.fullnet_ddp if self.IS_DISTRIBUTED else self.fullnet

        loss_train = []
        print("\x1b[s", end="") # VT100 save curse
        for item in dl:
            heatmap = None
            item_idx = 2

            # img_seq, K, gt = item[:item_idx]
            img_seq, K = item[:item_idx]

            if self.WITH_KPHEATMAP:
                heatmap = item[item_idx]
                item_idx += 1

            img_seq = img_seq.to(self.TENSOR_DEVICE)
            K = K.to(self.TENSOR_DEVICE)
            gt = None #gt.to(self.TENSOR_DEVICE)
            
            if heatmap is not None:
                heatmap = heatmap.to(self.TENSOR_DEVICE)

            rst = self.proc_one_batch(fullnet, img_seq, K, gt, heatmap, WITH_ACC_POSE=self.WITH_ACC_POSE, WITH_AUTOMASK=self.WITH_AUTOMASK)

            rst_ssim_loss = rst.rst_ssim_loss
            rst_l1_loss = rst.rst_l1_loss
            rst_l2_loss = rst.rst_l2_loss
            rst_disp_smooth_loss = rst.rst_disp_smooth_loss
            loss = rst.loss # type: torch.Tensor
            loss_raw = rst.loss_raw
            
            self.fullnet.zero_grad_optimizers()

            loss.backward()

            self.fullnet.step_optimizers()

            if self.IS_DISTRIBUTED:
                rst_ssim_loss = reduce_val(rst_ssim_loss)
                rst_l1_loss = reduce_val(rst_l1_loss)
                rst_l2_loss = reduce_val(rst_l2_loss)
                rst_disp_smooth_loss = reduce_val(rst_disp_smooth_loss)
                loss = reduce_val(loss)
                loss_raw = reduce_val(loss_raw)

                dist.barrier()

            # print losses
            if self.global_step % 20 == 0:
                self.print_losses({"ssim": rst_ssim_loss,
                                   "l1": rst_l1_loss,
                                   "l2": rst_l2_loss,
                                   "disp": rst_disp_smooth_loss,
                                   "loss": loss,
                                   "loss_raw": loss_raw})

            # record
            if self.global_step % 10 == 0:
                self.record_losses({"loss": loss,
                                    "ssim": rst_ssim_loss,
                                    "l1": rst_l1_loss,
                                    "l2": rst_l2_loss,
                                    "disp": rst_disp_smooth_loss,
                                    "loss_raw": loss_raw},
                                   self.global_step // 10)

            self.global_step += 1

            loss_train.append(loss.detach())

            pass
        # } for item in dl

        loss_train = torch.tensor(loss_train).mean()
        return loss_train

    def train(self):
        self.train_init()
        with Timer() as t:
            self.train_start()
            self.train_step(self.EPOCH_SIZE, t)
            self.train_end()
        pass

    def train_start(self):
        if not self.train_inited:
            self.train_init()

        print("program start.", flush=True)

        self.sw = SummaryWriter(self.path_sw_dir)

        self.fullnet.to(self.TENSOR_DEVICE)
        self.fullnet.train()

        if self.IS_DISTRIBUTED:
            self.fullnet = nn.SyncBatchNorm.convert_sync_batchnorm(self.fullnet)
            self.fullnet_ddp = DistributedDataParallel(self.fullnet, device_ids=[self.RANK], find_unused_parameters=True)
            self.fullnet_ddp.train()

        self.global_step = 0
        self.loss_val_min = 1000
        self.epoch_save_models = -1

        # check and resume
        if self.RESUME:
            print("resume")
            self.resume_models(self.TENSOR_DEVICE)
            self.resume_training_models(self.TENSOR_DEVICE)
            pass
        # }
        
        # save models
        print("saveing init models...", flush=True)
        self.save_models()
        self.save_training_models()
        print("init models saved.", flush=True)

        self.train_started = True

    def train_step(self, epoch_size, t=None):
        # type: (int, Timer) -> None
        if not self.train_started:
            self.train_start()

        # Iterative
        print("iterate {} epoches".format(epoch_size), flush=True)
        for epoch in range(epoch_size):
            if self.IS_DISTRIBUTED:
                self.sampler.set_epoch(epoch)
            loss_train = self.train_one_epoch(self.dl)
            loss_train = loss_train.cpu().item()
            self.sw.add_scalar("train/loss", loss_train, epoch)

            self.fullnet.step_lr_schedulers()


            
            print("validating ...")
            loss_val = self.eval_valid(self.dl_val)
            self.sw.add_scalar("val/loss", loss_val, epoch)
            print("val loss: {}".format(loss_val))
            print("======")

            # save models
            if loss_val < self.loss_val_min:
                print("saving epoch {}/{} model...".format(epoch + 1, epoch_size), flush=True)
                self.save_models()
                self.save_training_models()
                self.loss_val_min = loss_val
                self.epoch_save_models = epoch
            elif epoch - self.epoch_save_models >= 10: # 取消注释这段代码时最好把各个TrainingModule在save_models时保留的版本数设大一点。
                print("save epoch {}/{} model as 10 epoches passed from last save...".format(epoch + 1, epoch_size), flush=True)
                self.save_models()
                self.save_training_models()
                self.epoch_save_models = epoch
            else:
                print("loss of epoch {}/{} model in validation set is not smallest. No model was saved.".format(epoch + 1, epoch_size), flush=True)

            if t is not None:
                t.print_now()
            print("=================")

            pass
        # } for epoch in range(EPOCH_SIZE)
            
        # metrics = {"loss": loss.cpu().item(),
        #             "rst_ssim_loss": rst_ssim_loss.cpu().item(),
        #             "rst_l1_loss": rst_l1_loss.cpu().item(),
        #             "rst_l2_loss": rst_l2_loss.cpu().item(),
        #             "rst_disp_smooth_loss": rst_disp_smooth_loss.cpu().item()}
        # sw.add_hparams(self.hyperparams_dict, metrics)

    def train_end(self):
        if not self.train_started:
            return

        self.sw.close()

        self.train_started = False

        pass
    # } def train()

class ProcOneBatchResult:
    '''
    Similar to ProcOneBatchForwardResult, but the field of this class is one single loss, not a tensor with multiple dimensions. 
    Use 2 classes just for clearity of different method retval.
    '''
    def __init__(self, rst_ssim_loss, rst_l1_loss, rst_l2_loss, rst_disp_smooth_loss=None, loss=None, disp_reproj_loss=None, loss_raw=None, vae_loss=None, flow_loss=None, flow_w_reproj_loss=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> None
        '''
        Similar to ProcOneBatchForwardResult, but the field of this class is one single loss, not a tensor with multiple dimensions. 
        Use 2 classes just for clearity of different method retval.
        '''
        self.rst_ssim_loss = rst_ssim_loss
        self.rst_l1_loss = rst_l1_loss
        self.rst_l2_loss = rst_l2_loss
        self.rst_disp_smooth_loss = rst_disp_smooth_loss
        self.loss = loss
        self.disp_reproj_loss =disp_reproj_loss
        self.loss_raw = loss_raw
        self.vae_loss = vae_loss
        self.flow_loss = flow_loss
        self.flow_w_reproj_loss = flow_w_reproj_loss
    # }
# }

class ProcOneBatchForwardResult:
    '''
    similar to ProcOneBatchResult, but the field of this class is a tensor with multiple dimensions, not a single loss. 
    Use 2 classes just for clearity of different method retval.
    '''
    def __init__(self, reprojection_loss, rst_disp_smooth_loss, disp_reprojection_loss=None, rst_vae_loss=None, rst_flow_loss=None, rst_flow_w_reproj_loss=None):
        # type: (ComputeReprojectionLossAccResult, torch.Tensor, torch.Tensor, torch.Tensor, ComputeReprojectionLossResult, ComputeReprojectionLossResult) -> None
        '''
        similar to ProcOneBatchResult, but the field of this class is a tensor with multiple dimensions, not a single loss. 
        Use 2 classes just for clearity of different method retval.
        '''
        self.reprojection_loss = reprojection_loss
        self.rst_disp_smooth_loss = rst_disp_smooth_loss
        self.disp_reprojection_loss = disp_reprojection_loss
        self.rst_vae_loss = rst_vae_loss
        self.rst_flow_loss = rst_flow_loss
        self.rst_flow_w_reproj_loss = rst_flow_w_reproj_loss
    # }
# }

class ComputeReprojectionLossResult:
    '''
    similar to ProcOneBatchResult, but the field of this class is a tensor with multiple dimensions, not a single loss. 
    Use 2 classes just for clearity of different method retval.
    '''
    def __init__(self, rst_ssim_loss, rst_l1_loss, rst_l2_loss, mask=None, rst_ssim_loss_raw=None, rst_l1_loss_raw=None, rst_l2_loss_raw=None):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> None

        '''
        similar to ProcOneBatchResult, but the field of this class is a tensor with multiple dimensions, not a single loss. 
        Use 2 classes just for clearity of different method retval.
        '''
        self.rst_ssim_loss = rst_ssim_loss
        self.rst_l1_loss = rst_l1_loss
        self.rst_l2_loss = rst_l2_loss
        self.mask = mask
        self.rst_ssim_loss_raw = rst_ssim_loss_raw
        self.rst_l1_loss_raw = rst_l1_loss_raw
        self.rst_l2_loss_raw = rst_l2_loss_raw

    def deconstruct(self):
        return self.rst_ssim_loss, self.rst_l1_loss, self.rst_l2_loss, self.mask, self.rst_ssim_loss_raw, self.rst_l1_loss_raw, self.rst_l2_loss_raw
    # }
# } class ComputeReprojectionLossAccResult

class ComputeReprojectionLossAccResult:
    '''
    similar to ProcOneBatchResult, but the field of this class is a tensor with multiple dimensions, not a single loss. 
    Use 2 classes just for clearity of different method retval.
    '''
    def __init__(self, rst_ssim_loss, rst_l1_loss, rst_l2_loss, rst_coors_loss=None, rst_gt_reproj_ssim_loss=None, rst_ori_ssim=None, rst_ori_l1=None, rst_ori_l2=None, mask=None, rst_ssim_loss_raw=None, rst_l1_loss_raw=None, rst_l2_loss_raw=None):
        '''
        similar to ProcOneBatchResult, but the field of this class is a tensor with multiple dimensions, not a single loss. 
        Use 2 classes just for clearity of different method retval.
        '''
        self.rst_ssim_loss = rst_ssim_loss
        self.rst_l1_loss = rst_l1_loss
        self.rst_l2_loss = rst_l2_loss
        self.rst_coors_loss = rst_coors_loss
        self.rst_gt_reproj_ssim_loss = rst_gt_reproj_ssim_loss
        self.rst_ori_ssim = rst_ori_ssim
        self.rst_ori_l1 = rst_ori_l1
        self.rst_ori_l2 = rst_ori_l2
        self.mask = mask
        self.rst_ssim_loss_raw = rst_ssim_loss_raw
        self.rst_l1_loss_raw = rst_l1_loss_raw
        self.rst_l2_loss_raw = rst_l2_loss_raw
    # }
# } class ComputeReprojectionLossAccResult

def reduce_val(val):
    world_size = dist.get_world_size()
    with torch.no_grad():
        dist.all_reduce(val, async_op=True)
        val /= world_size
    return val
