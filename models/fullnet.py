# Based on DeepVO-pytorch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable
import os
import os.path
import abc
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional
import numpy as np
from .seq2batch import SeqToBatch
from .encoder2i import Encoder2i
from .depth_decoder1o import DepthDecoder1o
from .pose_decoder1o import PoseDecoder1o
from .resnet_encoder2i import ResnetEncoder2i
from utils.torchutils import seq_adjacent_concat
from . import TrainingModule

class FullnetBase(TrainingModule):
    def __init__(self, network_dict={}, **kwargs):
        # type: (Dict[str, TrainingModule], TrainingModule) -> None
        super().__init__()
        self.networks = {} # type: Dict[str, TrainingModule]
        self.train_network = {} # type: Dict[str, bool]
        self._append_networks(network_dict, **kwargs)

    def _append_networks(self, network_dict={}, **kwargs):
        # type: (Dict[str, TrainingModule], TrainingModule) -> None
        self.networks.update(network_dict)
        self.networks.update(kwargs)
        for k in self.networks.keys():
            if k not in self.train_network.keys():
                self.train_network[k] = True

    def _set_train_network(self, network_train={}, **kwargs):
        # type: (Dict[str, bool], bool) -> None
        self.train_network.update(network_train)
        self.train_network.update(kwargs)

    def set_records_path(self, path_records_dir):
        self.path_records_dir = path_records_dir
        self.set_models_path(path_records_dir)
        self.set_optimizers_path(path_records_dir)
        self.set_lr_schedulers_path(path_records_dir)

    def set_models_path(self, path_records_dir=None):
        if path_records_dir is None:
            path_records_dir = self.path_records_dir
        if path_records_dir is not None:
            for k, v in self.networks.items():
                save_path = os.path.join(path_records_dir, "{}.pth".format(k))
                v.set_model_path(save_path)

    def set_optimizers_path(self, path_records_dir):
        if path_records_dir is None:
            path_records_dir = self.path_records_dir
        if path_records_dir is not None:
            for k, v in self.networks.items():
                save_path = os.path.join(path_records_dir, "optimizer_{}.pth".format(k))
                v.set_optimizer_path(save_path)

    def set_lr_schedulers_path(self, path_records_dir):
        if path_records_dir is None:
            path_records_dir = self.path_records_dir
        if path_records_dir is not None:
            for k, v in self.networks.items():
                save_path = os.path.join(path_records_dir, "lr_scheduler_{}.pth".format(k))
                v.set_lr_scheduler_path(save_path)

    def resume_models(self, device=None, version=0):
        for v in self.networks.values():
            v.resume_model(device=device, version=version)

    def save_models(self):
        if not os.path.exists(self.path_records_dir):
            os.makedirs(self.path_records_dir)
        for v in self.networks.values():
            v.save_model()

    def resume_optimizers(self, device=None, version=0):
        for v in self.networks.values():
            v.resume_optimizer(device=device, version=version)

    def save_optimizers(self):
        if not os.path.exists(self.path_records_dir):
            os.makedirs(self.path_records_dir)
        for v in self.networks.values():
            v.save_optimizer()

    def resume_lr_schedulers(self, device=None, version=0):
        for v in self.networks.values():
            v.resume_lr_scheduler(device=device, version=version)

    def save_lr_schedulers(self):
        if not os.path.exists(self.path_records_dir):
            os.makedirs(self.path_records_dir)
        for v in self.networks.values():
            v.save_lr_scheduler()

    def step_optimizers(self):
        for k, v in self.networks.items():
            if self.train_network[k]:
                v.step_optimizer()

    def zero_grad_optimizers(self):
        for k, v in self.networks.items():
            if self.train_network[k]:
                v.zero_grad_optimizer()

    def step_lr_schedulers(self):
        for k, v in self.networks.items():
            if self.train_network[k]:
                v.step_lr_scheduler()

class Fullnet(FullnetBase):
    def __init__(self, depth_encoder, depth_decoder, pose_encoder, pose_decoder, de_preparer=None, dd_preparer=None, pe_preparer=None, pd_preparer=None, ret_postproc=None):
        # type: (Union[Encoder2i, ResnetEncoder2i], DepthDecoder1o, Union[Encoder2i, ResnetEncoder2i], PoseDecoder1o, Optional[Callable[[torch.Tensor], torch.Tensor]], Optional[Callable[[torch.Tensor], torch.Tensor]], Optional[Callable[[torch.Tensor], torch.Tensor]], Optional[Callable[[torch.Tensor], torch.Tensor]], Optional[Callable[[Dict], Dict]]) -> None
        super().__init__(depth_encoder=depth_encoder, depth_decoder=depth_decoder, pose_encoder=pose_encoder, pose_decoder=pose_decoder)

        self.depth_encoder = depth_encoder
        self.depth_decoder = depth_decoder
        self.pose_encoder = pose_encoder
        self.pose_decoder = pose_decoder
        self.de_preparer = de_preparer if de_preparer is not None else self.depth_encoder_preparer
        self.dd_preparer = dd_preparer if dd_preparer is not None else self.depth_decoder_preparer
        self.pe_preparer = pe_preparer if pe_preparer is not None else self.pose_encoder_preparer
        self.pd_preparer = pd_preparer if pd_preparer is not None else self.pose_decoder_preparer
        self.ret_postproc = ret_postproc if ret_postproc is not None else self.ret_post_processor

    def depth_encoder_preparer(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return x

    def depth_decoder_preparer(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return x

    def pose_encoder_preparer(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return seq_adjacent_concat(x)

    def pose_decoder_preparer(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        return x[-1]

    def ret_post_processor(self, x):
        # type: (Dict) -> Dict
        return x

    def forward(self, x): 
        '''
        ## Parameters:

        - x: (b, t, c3, h, w)
        '''
        # img_adj = seq_adjacent_concat(x)
        self.out_depth_encoder = self.depth_encoder(self.de_preparer(x)) # tuple[(b, t, c, h, w)]
        self.out_depth_decoder = self.depth_decoder(self.dd_preparer(self.out_depth_encoder)) # tuple[(b, t, 1, h, w)]
        self.out_pose_encoder = self.pose_encoder(self.pe_preparer(x)) # tuple[(b, t, c, h, w)]
        self.out_pose_decoder = self.pose_decoder(self.pd_preparer(self.out_pose_encoder)) # type: Tuple[torch.Tensor, torch.Tensor]

        ret = { "out_depth_decoder": self.out_depth_decoder,
                "out_pose_decoder": self.out_pose_decoder }

        return self.out_depth_decoder, self.out_pose_decoder, self.ret_postproc(ret)

    def get_posenet(self):
        return PoseNet(self.pose_encoder, self.pose_decoder)

    def get_depthnet(self):
        return DepthNet(self.depth_encoder, self.depth_decoder)

    def train_depth_encoder(self, train=True):
        self._set_train_network(depth_encoder=train)

    def train_depth_decoder(self, train=True):
        self._set_train_network(depth_decoder=train)

    def train_pose_encoder(self, train=True):
        self._set_train_network(pose_encoder=train)

    def train_pose_decoder(self, train=True):
        self._set_train_network(pose_decoder=train)

class PoseNet(FullnetBase): # means encoder with 2 inputs
    def __init__(self, pose_encoder, pose_decoder):
        # type: (Union[Encoder2i, ResnetEncoder2i], PoseDecoder1o) -> None
        super().__init__(pose_encoder=pose_encoder, pose_decoder=pose_decoder)

        self.pose_encoder = pose_encoder
        self.pose_decoder = pose_decoder

    def forward(self, x): 
        '''
        ## Parameters:

        - x: (b, t, c6, h, w)
        '''
        out_pose_encoder = self.pose_encoder(x) # tuple[(b, t, c, h, w)]
        out_pose_decoder = self.pose_decoder(out_pose_encoder[-1]) # type: Tuple[torch.Tensor, torch.Tensor]
        
        ret = { "out_pose_decoder": out_pose_decoder }

        return out_pose_decoder, ret

    def train_pose_encoder(self, train=True):
        self._set_train_network(pose_encoder=train)

    def train_pose_decoder(self, train=True):
        self._set_train_network(pose_decoder=train)

class DepthNet(FullnetBase): # means encoder with 2 inputs
    def __init__(self, depth_encoder, depth_decoder):
        # type: (Union[Encoder2i, ResnetEncoder2i], DepthDecoder1o) -> None
        super().__init__(depth_encoder=depth_encoder, depth_decoder=depth_decoder)

        self.depth_encoder = depth_encoder
        self.depth_decoder = depth_decoder

    def forward(self, x): 
        '''
        ## Parameters:

        - x: (b, t, c3, h, w)
        '''
        out_encoder = self.depth_encoder(x) # tuple[(b, t, c, h, w)]
        out_depth_decoder = self.depth_decoder(out_encoder) # tuple[(b, t, 1, h, w)]
        
        ret = { "out_depth_decoder": out_depth_decoder }

        return out_depth_decoder, ret

    def train_depth_encoder(self, train=True):
        self._set_train_network(depth_encoder=train)

    def train_depth_decoder(self, train=True):
        self._set_train_network(depth_decoder=train)

