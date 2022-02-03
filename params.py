import os
from enum import Enum
from torch import optim
from torch.optim import lr_scheduler

class DatasetNameEnum(Enum):
    KITTI_ODOM = "kitti_odom"

class Parameters():
    def __init__(self):
        self.not_cuda = False
        self.cuda_visible_devices = (3, 4, 5, 0, 1, 2)
        self.cudnn_backend = True
        self.is_distributed = 0
        self.config_name = "base"
        self.network = 1

        # Path
        self.ori_dataset_dir =  'dataset/kitti_odom/original'
        self.dataset_dir =  'dataset/kitti_odom/preprocessed'

        self.train_sequence = [0, 1, 2, 3, 4, 5, 6]
        self.valid_sequence = [7, 8]
        self.test_sequence = [9, 10]

        # Prepare
        self.split_tag = "no_static"
        self.dataset_related_dir = "dataset/kitti_odom/related"
        self.seq_length = 5
        self.remove_static = True
        self.interval = self.seq_length
        self.heatmap_radius = 16
        self.heatmap_kernel_func = "linear" # refer to dict HEATMAP_KERNEL_FUNCS in method gen_heatmap in file prepare.py
        self.keypoint = "SIFT"

        # Data Preprocessing
        self.img_width = 416 # 608   # original size is about 1226
        self.img_height = 128 # 184   # original size is about 370
        self.cut = None # None or (u, v, h, w)

        self.depth_min = 0.1
        self.depth_max = 80

        # Training
        self.epoch_size = 45
        self.batch_size = 3
        self.lr = .0001
        self.weight_decay = 0.
        self.betas_adam = (0.9, 0.999)
        self.lr_scheduler_gamma = 1.
        self.optimizer_creator = lambda x: optim.Adam(x, lr=self.lr, betas=self.betas_adam, weight_decay=self.weight_decay)
        self.optimizer_creator_for_depth_decoder = self.optimizer_creator
        self.optimizer_creator_for_depth_encoder = self.optimizer_creator
        self.optimizer_creator_for_pose_decoder = self.optimizer_creator
        self.optimizer_creator_for_pose_encoder = self.optimizer_creator
        self.optimizer_creator_for_vae_decoder = self.optimizer_creator
        self.lr_scheduler_creator = lambda x : lr_scheduler.StepLR(x, step_size=1, gamma=self.lr_scheduler_gamma)
        self.lr_scheduler_creator_for_depth_decoder = self.lr_scheduler_creator
        self.lr_scheduler_creator_for_depth_encoder = self.lr_scheduler_creator
        self.lr_scheduler_creator_for_pose_decoder = self.lr_scheduler_creator
        self.lr_scheduler_creator_for_pose_encoder = self.lr_scheduler_creator
        self.lr_scheduler_creator_for_vae_decoder = self.lr_scheduler_creator
        self.ssim_loss_factor = .85
        self.l1_loss_factor = 0.10
        self.l2_loss_factor = 0.05
        self.disp_smooth_loss_factor = 1.e-3

        self.WITH_AUTOMASK = False
        self.WITH_KPHEATMAP_WEIGHTING = True
        self.WITH_KPHEATMAP_MASK = False
        self.WITH_KPHEATMAP = self.WITH_KPHEATMAP_WEIGHTING or self.WITH_KPHEATMAP_MASK
        self.WITH_ACC_POSE = True
        self.WITH_AUG = False
        
        # Pretrain, Resume training
        self.pretrained_flownet = './pretrained/flownets_bn_EPE2.459.pth.tar'
                                # Choice:
                                # None
                                # './pretrained/flownets_bn_EPE2.459.pth.tar'
                                # './pretrained/flownets_EPE1.951.pth.tar'
        self.pretrained = True
        self.resume = False  # resume training
        self.comment = None
        self.dataset_name = DatasetNameEnum.KITTI_ODOM # type: DatasetNameEnum
        self.pretrained_2nd_root = ''

        # Test
        self.test_seq_id = 10
        self.test_frame_idx = 1
        self.test_version = 0

        self.debug = False
        
_params_dict = {}

def s1_param(par=None):
    if par is None:
        par = Parameters()

    par.network = 1
    par.depth_min = 0.1
    par.depth_max = 100
    par.epoch_size = 60
    par.batch_size = 3
    par.lr = .0001
    par.weight_decay = 0.
    par.betas_adam = (0.9, 0.999)
    par.lr_scheduler_gamma = .97
    par.optimizer_creator = lambda x: optim.Adam(x, lr=par.lr, betas=par.betas_adam, weight_decay=par.weight_decay)
    par.optimizer_creator_for_depth_decoder = par.optimizer_creator
    par.optimizer_creator_for_depth_encoder = par.optimizer_creator
    par.optimizer_creator_for_pose_decoder = par.optimizer_creator
    par.optimizer_creator_for_pose_encoder = par.optimizer_creator
    par.optimizer_creator_for_vae_decoder = par.optimizer_creator
    par.lr_scheduler_creator = lambda x : lr_scheduler.StepLR(x, step_size=1, gamma=par.lr_scheduler_gamma)
    par.lr_scheduler_creator_for_depth_decoder = par.lr_scheduler_creator
    par.lr_scheduler_creator_for_depth_encoder = par.lr_scheduler_creator
    par.lr_scheduler_creator_for_pose_decoder = par.lr_scheduler_creator
    par.lr_scheduler_creator_for_pose_encoder = par.lr_scheduler_creator
    par.lr_scheduler_creator_for_vae_decoder = par.lr_scheduler_creator
    par.ssim_loss_factor = .85
    par.l1_loss_factor = 0.10
    par.l2_loss_factor = 0.05
    par.disp_smooth_loss_factor = 1.e-3
    par.remove_outsite_pixel = False

    par.WITH_AUTOMASK = False
    par.WITH_KPHEATMAP_WEIGHTING = True
    par.WITH_KPHEATMAP_MASK = False
    par.WITH_KPHEATMAP = par.WITH_KPHEATMAP_WEIGHTING or par.WITH_KPHEATMAP_MASK
    par.WITH_ACC_POSE = True
    par.WITH_AUG = False
    
    return par
_params_dict["s1"] = s1_param

def base_param(par=None):
    if par is None:
        par = Parameters()
    return par
_params_dict["base"] = base_param

def seqlen3_param(par=None):
    if par is None:
        par = Parameters()

    par.split_tag = "no_static_l3"
    par.seq_length = 3

    return par
_params_dict["seqlen3"] = seqlen3_param

def debug_param(par=None):
    if par is None:
        par = Parameters()
    par.config_name = "debug"
    par.split_tag = "debug"
    par.epoch_size = 1
    par.debug = True

    par.train_sequence = [0]
    par.valid_sequence = [7]
    par.test_sequence = [9]

    return par
_params_dict["debug"] = debug_param

_param_ctor = base_param

param = _param_ctor()

def get_param(config_name = None):
    '''
    return a new Parameters object
    '''
    if config_name is not None:
        if isinstance(config_name, tuple) or isinstance(config_name, list):
            config = _param_ctor()
            for cfg in config_name:
                if cfg in _params_dict.keys():
                    config = _params_dict[cfg](config)
                else:
                    raise Exception("Config \"{}\" do not exist.".format(cfg))
            return config
        else:
            if config_name in _params_dict.keys():
                return _params_dict[config_name]()
            else:
                raise Exception("Config \"{}\" do not exist.".format(config_name))
    else:
        return _param_ctor()

    