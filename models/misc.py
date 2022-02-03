from typing import Iterable, Optional, Tuple, Union, Callable
import os
import os.path
import torch
from .resnet_encoder2i import ResnetEncoder2i
from .encoder2i import Encoder2i
from .depth_decoder1o import DepthDecoder1o
from .pose_decoder1o import PoseDecoder1o, PoseDecoder1o2
from .fullnet import Fullnet, PoseNet, DepthNet, FullnetBase
from params import Parameters

def network_enc_depdec_posdec_creator(param, pretrained=True, img_height=128, img_width=416, init_optim_and_lr_sched=True): # 0
    # type: (Parameters, bool, int, int, bool) -> FullnetBase
    depth_encoder = Encoder2i(batch_first=True, inplane=3, pretrained=param.pretrained_flownet)
    pose_encoder = Encoder2i(batch_first=True, inplane=6, pretrained=param.pretrained_flownet)
    upplanes = [i for i in depth_encoder.planes]
    upplanes.reverse()
    upplanes.append(16)
    depth_decoder = DepthDecoder1o(inplanes=depth_encoder.planes, upplanes=upplanes, output_size=(img_height, img_width))
    pose_decoder = PoseDecoder1o(inplane=pose_encoder.planes[-1])

    if init_optim_and_lr_sched:
        depth_encoder.create_optimizer(param.optimizer_creator_for_depth_encoder)
        depth_decoder.create_optimizer(param.optimizer_creator_for_depth_decoder)
        pose_encoder.create_optimizer(param.optimizer_creator_for_pose_encoder)
        pose_decoder.create_optimizer(param.optimizer_creator_for_pose_decoder)

        depth_encoder.create_lr_scheduler(param.lr_scheduler_creator_for_depth_encoder)
        depth_decoder.create_lr_scheduler(param.lr_scheduler_creator_for_depth_decoder)
        pose_encoder.create_lr_scheduler(param.lr_scheduler_creator_for_pose_encoder)
        pose_decoder.create_lr_scheduler(param.lr_scheduler_creator_for_pose_decoder)

    return Fullnet(depth_encoder, depth_decoder, pose_encoder, pose_decoder)

def network_resenc_depdec_posdec_creator(param, pretrained=True, img_height=128, img_width=416, init_optim_and_lr_sched=True, resnet_layers=50): # 1
    # type: (Parameters, bool, int, int, bool, int) -> FullnetBase
    depth_encoder = ResnetEncoder2i(num_layers=resnet_layers, pretrained=bool(pretrained), inplane=3, use_convlstm=False)
    pose_encoder = ResnetEncoder2i(num_layers=resnet_layers, pretrained=bool(pretrained), inplane=6, use_convlstm=False)
    upplanes = [i for i in depth_encoder.planes]
    upplanes.reverse()
    upplanes.append(16)
    depth_decoder = DepthDecoder1o(inplanes=depth_encoder.planes, upplanes=upplanes, output_size=(img_height, img_width))
    pose_decoder = PoseDecoder1o(inplane=pose_encoder.planes[-1])

    if init_optim_and_lr_sched:
        depth_encoder.create_optimizer(param.optimizer_creator_for_depth_encoder)
        depth_decoder.create_optimizer(param.optimizer_creator_for_depth_decoder)
        pose_encoder.create_optimizer(param.optimizer_creator_for_pose_encoder)
        pose_decoder.create_optimizer(param.optimizer_creator_for_pose_decoder)

        depth_encoder.create_lr_scheduler(param.lr_scheduler_creator_for_depth_encoder)
        depth_decoder.create_lr_scheduler(param.lr_scheduler_creator_for_depth_decoder)
        pose_encoder.create_lr_scheduler(param.lr_scheduler_creator_for_pose_encoder)
        pose_decoder.create_lr_scheduler(param.lr_scheduler_creator_for_pose_decoder)
    return Fullnet(depth_encoder, depth_decoder, pose_encoder, pose_decoder)

NETWORK_DICT = (network_enc_depdec_posdec_creator, # 0
                network_resenc_depdec_posdec_creator, # 1
                )

def create_networks(network, param, pretrained=True, img_height=128, img_width=416, init_optim_and_lr_sched=True):
    """
    ## Parameter:

     - network
     | network | Depth | Pose | note |
     |---------|:------|:-----|------|
     | 0 | Enc + DepthDec | Enc + PoseDec | None |
     | 1 | ResNetEnc + DepthDec | ResNetEnc + PoseDec | None |
    """
    NETWORK_CREATOR = NETWORK_DICT[network]
    return NETWORK_CREATOR(param, pretrained, img_height, img_width, init_optim_and_lr_sched)


def create_pose_networks(network, param, pretrained=True, img_height=128, img_width=416, init_optim_and_lr_sched=True):
    # create networks
    fullnet = create_networks(network, param, pretrained, img_height, img_width, init_optim_and_lr_sched)

    return fullnet.get_posenet()

def create_depth_networks(network, param, pretrained=True, img_height=128, img_width=416, init_optim_and_lr_sched=True):
    # create networks
    fullnet = create_networks(network, param, pretrained, img_height, img_width, init_optim_and_lr_sched)

    return fullnet.get_depthnet()
