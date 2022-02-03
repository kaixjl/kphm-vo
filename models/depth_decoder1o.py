# Based on GeoNet-PyTorch

from collections import OrderedDict
from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_
from . import TrainingModule


def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


def crop_like(input, ref):
    if ref is None:
        return input
    if isinstance(ref, torch.Tensor):
        assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
        return input[:, :, :ref.size(2), :ref.size(3)]
    elif isinstance(ref, tuple) or isinstance(ref, list):
        assert(input.size(2) >= ref[0] and input.size(3) >= ref[1])
        return input[:, :, :ref[0], :ref[1]]



class DepthDecoder1o(TrainingModule):

    def __init__(self, inplanes = [64, 128, 256, 512, 512, 1024], upplanes = [1024, 512, 512, 256, 128, 64, 16], alpha=10, beta=0.01, output_size = (128, 416), predict_disp_len = 3):
        '''

        '''
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.output_size = output_size

        conv_planes = inplanes # [64, 128, 256, 512, 512, 1024]

        upconv_planes = upplanes # [1024, 512, 512, 256, 128, 64, 16]
        self.conv_planes = conv_planes
        self.upconv_planes = upconv_planes

        assert(conv_planes[-1]==upconv_planes[0])
        assert(len(conv_planes) + 1 <= len(upconv_planes))

        # upconvs
        self.upconvs = nn.ModuleDict()
        len_upplanes_1 = len(upconv_planes) - 1
        for i in range(len_upplanes_1):
            module_name = "upconv{}".format(len_upplanes_1-i)
            module_added = upconv(upconv_planes[i], upconv_planes[i+1])
            self.upconvs.add_module(module_name, module_added)

        self.predict_disp_len = predict_disp_len
        
        # iconvs
        self.iconvs = nn.ModuleDict()
        for i in range(len_upplanes_1): # 0 .. len(upconv_planes) - 1
            idx_upconv = i + 1 # 1 .. = len(upconv_planes) - 1
            idx_conv =  len(conv_planes) - 2 - i # len(conv_planes) - 2 .. -1 ..
            idx_name = len_upplanes_1 - i # len(upconv_planes) - 1 .. -1 .. =1
            module_name = "iconv{}".format(idx_name)


            module_added = conv(upconv_planes[idx_upconv] + (conv_planes[idx_conv] if idx_conv >=0 else 0), upconv_planes[idx_upconv])
            # module_added = conv((0 if idx_name >= predict_disp_len else 1) + upconv_planes[idx_upconv] + (conv_planes[idx_conv] if idx_conv >=0 else 0), upconv_planes[idx_upconv])
            self.iconvs.add_module(module_name, module_added)

        # predict_disps
        self.predict_disps = nn.ModuleDict()
        for i in range(predict_disp_len):
            module_name = "predict_disp{}".format(predict_disp_len - i)
            module_added = predict_disp(upconv_planes[len(upconv_planes) - predict_disp_len + i])
            self.predict_disps.add_module(module_name, module_added)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # type: (Union[List[torch.Tensor], Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]
        '''
        ## Parameters:
        
        - x: [out_conv1, out_conv2, ..., bottleneck]

        ## Return:

        tuple [Tensor(b, t, c, H, W), Tensor(b, t, c, H //2, W // 2), ..., lower resolution]. Tuple of disparity from layer -1 (highest resolution), -2, ..., to lowest resolution. (b, t, 1, h, w)
        '''
        out_conv = x
        shapes = [t.shape for t in out_conv]
        out_conv = [t.reshape((-1,)+s[2:]) for t, s in zip(out_conv, shapes)]

        disps = OrderedDict()
        len_upplanes_1 = len(self.upconv_planes) - 1
        out_iconv = out_conv[-1]
        for i in range(len_upplanes_1, 0, -1): # len(upconv_planes) - 1 .. -1 ..
            idx_name = i # len(upconv_planes) -1 .. -1 .. =1
            idx_conv_planes = i - (len_upplanes_1 - len(self.conv_planes)) - 2 # len(conv_planes) - 2 .. -1 ..
            idx_out_conv = i - (len_upplanes_1 - len(out_conv)) - 2 # len(conv_planes) - 2 .. -1 ..

            out_upconv = crop_like(self.upconvs["upconv{}".format(idx_name)](out_iconv), out_conv[idx_out_conv] if idx_out_conv >= 0 else (self.output_size if idx_name == 1 else None))
            concatenatee = (out_upconv,)

            if idx_conv_planes >= 0:
                concatenatee += (out_conv[idx_out_conv],)

            # if idx_name <= self.predict_disp_len - 1: # self.predict_disp_len - 1 .. -1 .. =1
            #     disp_up = crop_like(F.interpolate(disp, scale_factor=2, mode='bilinear', align_corners=False), out_conv[idx_out_conv] if idx_out_conv >= 0 else self.output_size)
            #     concatenatee += (disp_up,)

            concat = torch.cat(concatenatee, 1)
            out_iconv = self.iconvs["iconv{}".format(idx_name)](concat)

            if idx_name <= self.predict_disp_len: # self.predict_disp_len .. -1 .. =1
                disp = self.alpha * self.predict_disps["predict_disp{}".format(idx_name)](out_iconv) + self.beta
                disps["disp{}".format(idx_name)] = disp
            

        return tuple(t.reshape(shapes[0][:2]+t.shape[1:]) for t in list(disps.values())[::-1])
