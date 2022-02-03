# Based on Monodepth2

from __future__ import absolute_import, division, print_function
from typing import Tuple

import torch
import torch.nn as nn
from collections import OrderedDict
from . import TrainingModule


class PoseDecoder1o(TrainingModule):
    CONV_2D = 0
    CONV_1D = 1
    def __init__(self, inplane = 1024, stride=1, conv_model=CONV_2D, act_func=nn.ReLU):
        super().__init__()

        self.conv_model = conv_model
        if conv_model==self.CONV_2D:
            conv_ = nn.Conv2d
        elif conv_model==self.CONV_1D:
            conv_ = nn.Conv1d

        self.inplane = inplane

        self.convs = nn.ModuleDict()
        self.convs.add_module("pose0", conv_(inplane, 256, 1))
        self.convs.add_module("pose1", conv_(256, 256, 3, stride, 1))
        self.convs.add_module("pose2", conv_(256, 256, 3, stride, 1))
        self.convs.add_module("pose3", conv_(256, 6, 1))

        self.relu0 = act_func()
        self.relu1 = act_func()
        self.relu2 = act_func()


    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        '''
        ## Parameters:

        - x: (b, t, c, h, w)

        ## Return:

        tuple with axisangle and translation.
        axisangle from the latter to the former.
        translation from the latter to the former.
        '''
        # last_features = [f[-1] for f in x]

        # cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        # cat_features = torch.cat(cat_features, 1)

        # out = cat_features
        # for i in range(3):
        #     out = self.convs[("pose", i)](out)
        #     if i != 2:
        #         out = self.relu(out)

        shapes = x.shape
        x = x.reshape((-1,) + shapes[2:])

        if self.conv_model==self.CONV_2D:
            out = x
        elif self.conv_model==self.CONV_1D:
            out = x.squeeze(-2)

        out = self.convs["pose0"](out)
        out = self.relu0(out)
        out = self.convs["pose1"](out)
        out = self.relu1(out)
        out = self.convs["pose2"](out)
        out = self.relu2(out)
        out = self.convs["pose3"](out)

        if self.conv_model==self.CONV_2D:
            out = out.mean(3).mean(2)
        elif self.conv_model==self.CONV_1D:
            out = out.mean(2)

        out = 0.01 * out

        out = out.reshape((shapes[:2] + out.shape[1:]))

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

class PoseDecoder1o2(TrainingModule):
    CONV_2D = 0
    CONV_1D = 1
    def __init__(self, inplane = 1024, stride=1, n_layer=4, conv_model=CONV_2D, act_func=nn.ReLU):
        '''
        ## Parameter
        
        - n_layer: >=2
        '''
        assert(n_layer>=2)
        super().__init__()

        self.conv_model = conv_model
        if conv_model==self.CONV_2D:
            conv_ = nn.Conv2d
        elif conv_model==self.CONV_1D:
            conv_ = nn.Conv1d

        self.inplane = inplane
        self.n_layer = n_layer

        self.convs = nn.ModuleDict()
        self.convs.add_module("pose0", conv_(inplane, 256, 1))
        for i in range(1, n_layer-1):
            self.convs.add_module("pose{}".format(i), conv_(256, 256, 3, stride, 1))
        self.convs.add_module("pose{}".format(n_layer-1), conv_(256, 6, 1))


        self.relu = act_func()


    def forward(self, x):
        # type: (torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        '''
        ## Parameters:

        - x: (b, t, c, h, w)

        ## Return:

        tuple with axisangle and translation.
        axisangle from the latter to the former.
        translation from the latter to the former.
        '''

        shapes = x.shape
        x = x.reshape((-1,) + shapes[2:])

        if self.conv_model==self.CONV_2D:
            out = x
        elif self.conv_model==self.CONV_1D:
            out = x.squeeze(-2)

        for i in range(self.n_layer-1):
            out = self.convs["pose{}".format(i)](out)
            out = self.relu(out)

        out = self.convs["pose{}".format(self.n_layer-1)](out)

        if self.conv_model==self.CONV_2D:
            out = out.mean(3).mean(2)
        elif self.conv_model==self.CONV_1D:
            out = out.mean(2)

        out = 0.01 * out

        out = out.reshape((shapes[:2] + out.shape[1:]))

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
