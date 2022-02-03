# Based on Monodepth2

from __future__ import absolute_import, division, print_function
from typing import List

import numpy as np

import torch
import torch.nn as nn
from torch.nn.modules import activation
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from .convlstm1 import ConvLSTM, ConvLSTMCell
from .seq2batch import SeqToBatch
from . import TrainingModule
from . import conv, convlstm

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(models.resnet.BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, output_activation=True):

        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)

        self.outout_activation = output_activation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.outout_activation:
            out = self.relu(out)

        return out


class Bottleneck(models.resnet.Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, output_activation=True):

        super().__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)

        self.outout_activation = output_activation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.outout_activation:
            out = self.relu(out)

        return out

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, inplane=3, use_convlstm=False, batch_first=False, output_activation=True, tail=False):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            inplane, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if tail:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            if use_convlstm:
                self.convlstm1 = convlstm(True, 64*block.expansion, 64*block.expansion, batch_first=batch_first)
                self.convlstm2 = convlstm(True, 128*block.expansion, 128*block.expansion, batch_first=batch_first)
                self.convlstm3 = convlstm(True, 256*block.expansion, 256*block.expansion, batch_first=batch_first)
                self.convlstm4 = convlstm(True, 512*block.expansion, 512*block.expansion, batch_first=batch_first)
            self.tail = nn.Sequential()
            self.tail.add_module("conv5", nn.Conv2d(
                512*block.expansion, 1024*block.expansion, kernel_size=3, stride=2, padding=1, bias=False))
            self.tail.add_module("bn5", nn.BatchNorm2d(1024*block.expansion))
            if output_activation:
                self.tail.add_module("relu5", nn.ReLU(inplace=True))
        else:
            if use_convlstm:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                self.convlstm1 = convlstm(True, 64*block.expansion, 64*block.expansion, batch_first=batch_first)
                self.convlstm2 = convlstm(True, 128*block.expansion, 128*block.expansion, batch_first=batch_first)
                self.convlstm3 = convlstm(True, 256*block.expansion, 256*block.expansion, batch_first=batch_first)
                self.convlstm4 = convlstm(True, 512*block.expansion, 512*block.expansion, batch_first=batch_first, activation=output_activation)
            else:
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2, output_activation=output_activation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, output_activation=True):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, output_activation=output_activation if i == blocks - 1 else True))

        return nn.Sequential(*layers)


def resnet_multiimage_input(num_layers, pretrained=False, inplane=3, use_convlstm=False, batch_first=False, output_activation=True, tail=False):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: BasicBlock, 50: Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, inplane=inplane, use_convlstm=use_convlstm, batch_first=batch_first, output_activation=output_activation, tail=tail)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)]) # type: dict
        # loaded['conv1.weight'] = torch.cat(
        #     [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        if inplane%3==0:
            num_input_images = inplane // 3
            loaded['conv1.weight'] = torch.cat(
                [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        else:
            new_conv1_weight = loaded['conv1.weight']
            new_conv1_weight = torch.cat([new_conv1_weight] * (inplane // 3) + [new_conv1_weight[:,0:inplane%3,:,:]], dim=1)
            loaded['conv1.weight'] = new_conv1_weight

        sd = model.state_dict()
        sd.update(loaded)
        model.load_state_dict(sd)
    return model


class ResnetEncoder2i(TrainingModule):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, inplane=6, use_convlstm=False, batch_first=False, output_activation=True, tail=False):
        super().__init__()
        if tail:
            self.planes = np.array([64, 64, 128, 256, 512, 1024])
        else:
            self.planes = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        # if num_input_images > 1:
        self.encoder = resnet_multiimage_input(num_layers, pretrained, inplane, use_convlstm=use_convlstm, batch_first=batch_first, output_activation=output_activation)
        # else:
            # self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.planes[1:] *= 4

        self.use_convlstm = use_convlstm
        self.tail = tail

    def forward(self, input_image):
        # type: (torch.Tensor) -> List[torch.Tensor]
        '''
        ## Parameters:

        - input_image: (b, t, c, h, w) or (t, b, c, h, w)

        ## Return:

        list [Tensor(b, t, c, H, W), Tensor(b, t, c, H //2, W // 2), ..., bottleneck], from front to end
        '''
        # converted from (b, t, c, h, w) (or (t, b, *)) to (N, c, h, w)
        shapes = input_image.shape
        input_image = input_image.flatten(0, 1)

        self.features = [] # type: List[torch.Tensor]
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        if self.use_convlstm:
            out = self.encoder.layer1(self.encoder.maxpool(self.features[-1]))
            out = out.reshape(shapes[:2]+out.shape[1:])
            out = self.encoder.convlstm1(out)
            self.features.append(out)

            out = self.encoder.layer2(self.features[-1])
            out = out.reshape(shapes[:2]+out.shape[1:])
            out = self.encoder.convlstm2(out)
            self.features.append(out)

            out = self.encoder.layer3(self.features[-1])
            out = out.reshape(shapes[:2]+out.shape[1:])
            out = self.encoder.convlstm3(out)
            self.features.append(out)

            out = self.encoder.layer4(self.features[-1])
            out = out.reshape(shapes[:2]+out.shape[1:])
            out = self.encoder.convlstm4(out)
            self.features.append(out)

            if self.tail:
                out = self.encoder.tail(out)
                self.features.append(out)
        else:
            self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
            self.features.append(self.encoder.layer2(self.features[-1]))
            self.features.append(self.encoder.layer3(self.features[-1]))
            self.features.append(self.encoder.layer4(self.features[-1]))
            if self.tail:
                self.features.append(self.encoder.tail(self.features[-1]))

        # converted back before returned
        return [f.reshape(shapes[:2] + f.shape[1:]) for f in self.features]
