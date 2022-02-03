# Based on DeepVO-pytorch
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional
import torch.optim
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from .convlstm1 import ConvLSTM, ConvLSTMCell
from utils.torchutils import seq_adjacent_concat, get_flownet_update_dict
from . import TrainingModule
from . import conv, convlstm

class Encoder2i(TrainingModule): # means encoder with 2 inputs

    planes = [64, 128, 256, 512, 512, 1024]
    
    def __init__(self, batchNorm=True, batch_first=False, inplane=6, output_activation=True, pretrained=False):
        super().__init__()
        # CNN
        self.batchNorm = batchNorm
        # self.clip = param.clip
        
        # b40c6w608h184 128 416
        conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
        # self.planes = [64, 128, 256, 512, 512, 1024]
        self.conv1   = conv(self.batchNorm,   inplane,   self.planes[0], kernel_size=7, stride=2, dropout=conv_dropout[0])# b40c64w304h92 64 208
        self.convlstm1_1 = convlstm(self.batchNorm, self.planes[0],  self.planes[0], kernel_size=3, dropout=conv_dropout[0], batch_first=batch_first)
        self.conv2   = conv(self.batchNorm,  self.planes[0],  self.planes[1], kernel_size=5, stride=2, dropout=conv_dropout[1])# b40c128w152h46 32 104   
        self.convlstm2_1 = convlstm(self.batchNorm, self.planes[1],  self.planes[1], kernel_size=3, dropout=conv_dropout[1], batch_first=batch_first) 
        self.conv3   = conv(self.batchNorm, self.planes[1],  self.planes[2], kernel_size=5, stride=2, dropout=conv_dropout[2])# b40c256w76h23 16 52
        self.convlstm3_1 = convlstm(self.batchNorm, self.planes[2],  self.planes[2], kernel_size=3, dropout=conv_dropout[3], batch_first=batch_first)# b40c256w76h23 16 52
        self.conv4   = conv(self.batchNorm, self.planes[2],  self.planes[3], kernel_size=3, stride=2, dropout=conv_dropout[4])# b40c512w38h12 8 26
        self.convlstm4_1 = convlstm(self.batchNorm, self.planes[3],  self.planes[3], kernel_size=3, dropout=conv_dropout[5], batch_first=batch_first)# b40c512w38h12 8 26
        self.conv5   = conv(self.batchNorm, self.planes[3],  self.planes[4], kernel_size=3, stride=2, dropout=conv_dropout[6])# b40c512w19h6 4 13
        self.convlstm5_1 = convlstm(self.batchNorm, self.planes[4],  self.planes[4], kernel_size=3, dropout=conv_dropout[7], batch_first=batch_first)# b40c512w19h6 4 13
        self.conv6   = conv(self.batchNorm, self.planes[4], self.planes[5], kernel_size=3, stride=2, dropout=conv_dropout[8])# b40c1024w10h3 
        self.convlstm6_1 = convlstm(self.batchNorm, self.planes[5],  self.planes[5], kernel_size=3, dropout=conv_dropout[3], batch_first=batch_first, activation=output_activation)

        # # Comput the shape based on diff image size
        # __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        # __tmp = self.encode_image(__tmp)

        # # RNN
        # self.rnn = nn.LSTM(
        #             input_size=int(np.prod(__tmp.size())), 
        #             hidden_size=param.rnn_hidden_size, 
        #             num_layers=2, 
        #             dropout=param.rnn_dropout_between, 
        #             batch_first=True)
        # self.rnn_drop_out = nn.Dropout(param.rnn_dropout_out)
        # self.linear = nn.Linear(in_features=param.rnn_hidden_size, out_features=6)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            # elif isinstance(m, nn.LSTM):
            #     # layer 1
            #     kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
            #     kaiming_normal_(m.weight_hh_l0)
            #     m.bias_ih_l0.data.zero_()
            #     m.bias_hh_l0.data.zero_()
            #     # Set forget gate bias to 1 (remember)
            #     n = m.bias_hh_l0.size(0)
            #     start, end = n//4, n//2
            #     m.bias_hh_l0.data[start:end].fill_(1.)

            #     # layer 2
            #     kaiming_normal_(m.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
            #     kaiming_normal_(m.weight_hh_l1)
            #     m.bias_ih_l1.data.zero_()
            #     m.bias_hh_l1.data.zero_()
            #     n = m.bias_hh_l1.size(0)
            #     start, end = n//4, n//2
            #     m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained is not None:
            self.load_pretrained_flownet(pretrained, torch.device("cpu"))

    def load_pretrained_flownet(self, path_model, device):
        pretrained_encoder_dict = torch.load(path_model, map_location=device)

        parameters_dict = self.state_dict()
        update_dict = get_flownet_update_dict(parameters_dict, pretrained_encoder_dict)
        parameters_dict.update(update_dict)
        self.load_state_dict(parameters_dict)

    def forward(self, x): 
        '''
        ## Parameters:

        - x: (b, t, c, h, w) or (t, b, c6, h, w)

        ## Return:

        list [Tensor(b, t, c, H, W), Tensor(b, t, c, H //2, W // 2), ..., bottleneck], from front to end
        '''

        # stack_image # b8s6c3w608h184
        # x = seq_adjacent_concat(x) # b8s5c6w608h184 # 构造encoder输入两两一组b(b×Cq2)c6hw
        self.input_shape = x.shape
        # CNN

        x = x - 0.5
        # b40c6w608h184
        x = self.encode_image(x) # b40c1024w10h3
        
        # x = x.view(batch_size, seq_len, -1) # b8s5l30720
        out = x


        # # RNN
        # out, hc = self.rnn(x) # b8s5l1000
        # out = self.rnn_drop_out(out) # b8s5l1000
        # out = self.linear(out) # b8s5l6
        return self.out_conv1, self.out_conv2, self.out_conv3, self.out_conv4, self.out_conv5, self.out_conv6
        

    def encode_image(self, x):
        # b40c6w608h184
        x = x.reshape((-1,) + self.input_shape[2:])
        self.out_conv1_0 = self.conv1(x) # b40c64w304h92
        self.out_conv1_0 = self.out_conv1_0.reshape(self.input_shape[:2]+self.out_conv1_0.shape[1:])
        self.out_conv1 = self.convlstm1_1(self.out_conv1_0)
        self.out_conv2_0 = self.conv2(self.out_conv1) # b40c128w152h46
        self.out_conv2_0 = self.out_conv2_0.reshape(self.input_shape[:2]+self.out_conv2_0.shape[1:])
        self.out_conv2 = self.convlstm2_1(self.out_conv2_0)
        self.out_conv3_0 = self.conv3(self.out_conv2) # b40c256w76h23
        self.out_conv3_0 = self.out_conv3_0.reshape(self.input_shape[:2]+self.out_conv3_0.shape[1:])
        self.out_conv3 = self.convlstm3_1(self.out_conv3_0) # b40c256w76h23
        self.out_conv4_0 = self.conv4(self.out_conv3) # b40c512w38h12
        self.out_conv4_0 = self.out_conv4_0.reshape(self.input_shape[:2]+self.out_conv4_0.shape[1:])
        self.out_conv4 = self.convlstm4_1(self.out_conv4_0) # b40c512w38h12
        self.out_conv5_0 = self.conv5(self.out_conv4) # b40c512w19h6
        self.out_conv5_0 = self.out_conv5_0.reshape(self.input_shape[:2]+self.out_conv5_0.shape[1:])
        self.out_conv5 = self.convlstm5_1(self.out_conv5_0) # b40c512w19h6
        self.out_conv6_0 = self.conv6(self.out_conv5) # b40c1024w10h3
        self.out_conv6_0 = self.out_conv6_0.reshape(self.input_shape[:2]+self.out_conv6_0.shape[1:])
        self.out_conv6 = self.convlstm6_1(self.out_conv6_0)

        def batch_to_seq(x):
            return x.reshape(self.input_shape[:2] + x.shape[1:])
            
        self.out_conv1 = batch_to_seq(self.out_conv1)
        self.out_conv2 = batch_to_seq(self.out_conv2)
        self.out_conv3 = batch_to_seq(self.out_conv3)
        self.out_conv4 = batch_to_seq(self.out_conv4)
        self.out_conv5 = batch_to_seq(self.out_conv5)
        self.out_conv6 = batch_to_seq(self.out_conv6)

        return self.out_conv6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    # def get_loss(self, x, y):
    #     predicted = self.forward(x)
    #     y = y[:, 1:, :]  # (batch, seq, dim_pose)
    #     # Weighted MSE Loss
    #     angle_loss = torch.nn.functional.mse_loss(predicted[:,:,:3], y[:,:,:3])
    #     translation_loss = torch.nn.functional.mse_loss(predicted[:,:,3:], y[:,:,3:])
    #     loss = (100 * angle_loss + translation_loss)
    #     return loss

    # def step(self, x, y, optimizer: torch.optim.Optimizer):
    #     optimizer.zero_grad()
    #     loss = self.get_loss(x, y)
    #     loss.backward()
    #     # if self.clip != None:
    #     #     torch.nn.utils.clip_grad_norm(self.rnn.parameters(), self.clip)
    #     optimizer.step()
    #     return loss

    def reset_convlstm_hidden_state(self):
        for m in self.modules():
            if isinstance(m, ConvLSTM):
                m.reset_hidden_state()
