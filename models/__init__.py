import os
import os.path
import abc
from typing import Callable, List, Union
import shutil
import torch
from torch import nn
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from .convlstm1 import ConvLSTM, ConvLSTMCell
from .seq2batch import SeqToBatch

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0, activation=True):
    layers = []
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False))
    if batchNorm:
        layers.append(nn.BatchNorm2d(out_planes))
    if activation:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    if dropout != 0:
        layers.append(nn.Dropout(dropout)) #, inplace=True)
    return nn.Sequential(*layers)

def convlstm(batchNorm, in_planes, out_planes, kernel_size=3, dropout=0, batch_first=False, activation=True):
    layers = []
    layers.append(ConvLSTM(in_planes, out_planes, kernel_size=kernel_size, num_layers=1, batch_first=batch_first, bias=False))
    layers.append(SeqToBatch())
    if batchNorm:
        layers.append(nn.BatchNorm2d(out_planes))
    if activation:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    if dropout != 0:
        layers.append(nn.Dropout(dropout)) #, inplace=True)
    return nn.Sequential(*layers)

class TrainingModule(nn.Module):
    def __init__(self, max_version=5) -> None:
        super().__init__()
        self.optimizer = None
        self.optimizer_save_path = None
        self.lr_scheduler = None
        self.lr_scheduler_save_path = None
        self.model_save_path = None
        self.max_version = max_version

    def load_pretrained(self, path_model, device=None):
        pretrained_dict = torch.load(path_model, map_location=device)

        parameters_dict = self.state_dict()
        parameters_dict.update({k:v for k,v in pretrained_dict.items() if k in parameters_dict.keys()})
        self.load_state_dict(parameters_dict)

    def set_optimizer(self, optimizer=None):
        # type: (Optimizer) -> None
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler=None):
        # type: (_LRScheduler) -> None
        self.lr_scheduler = lr_scheduler

    def create_optimizer(self, creator):
        # type: (Callable[[nn.Parameter], Optimizer]) -> None
        self.set_optimizer(creator(self.parameters()))

    def create_lr_scheduler(self, creator):
        # type: (Callable[[Optimizer], _LRScheduler]) -> None
        self.set_lr_scheduler(creator(self.optimizer))

    def set_model_path(self, save_path):
        self.model_save_path = save_path

    def set_optimizer_path(self, save_path):
        self.optimizer_save_path = save_path

    def set_lr_scheduler_path(self, save_path):
        self.lr_scheduler_save_path = save_path

    def resume_model(self, path=None, device=None, version=0):
        if path is None:
            path = self.model_save_path
        if path is not None:
            if version!=0:
                path = "{}.{}".format(path, version)
            self.load_state_dict(torch.load(path, map_location=device))

    def save_model(self, path=None):
        if path is None:
            path = self.model_save_path
        if path is not None:
            dir_name = os.path.dirname(path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            self.version_control_rolling_file(path, 0, self.max_version)
            torch.save(self.state_dict(), path)

    def resume_optimizer(self, path=None, device=None, version=0):
        if path is None:
            path = self.optimizer_save_path
        if path is not None:
            if version!=0:
                path = "{}.{}".format(path, version)
            self.optimizer.load_state_dict(torch.load(path, map_location=device))

    def save_optimizer(self, path=None):
        if path is None:
            path = self.optimizer_save_path
        if path is not None and self.optimizer is not None:
            dir_name = os.path.dirname(path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            self.version_control_rolling_file(path, 0, self.max_version)
            torch.save(self.optimizer.state_dict(), path)

    def resume_lr_scheduler(self, path=None, device=None, version=0):
        if path is None:
            path = self.lr_scheduler_save_path
        if path is not None:
            if version!=0:
                path = "{}.{}".format(path, version)
            self.lr_scheduler.load_state_dict(torch.load(path, map_location=device))

    def save_lr_scheduler(self, path=None):
        if path is None:
            path = self.lr_scheduler_save_path
        if path is not None and self.lr_scheduler is not None:
            dir_name = os.path.dirname(path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            self.version_control_rolling_file(path, 0, self.max_version)
            torch.save(self.lr_scheduler.state_dict(), path)

    def step_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def zero_grad_optimizer(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step_lr_scheduler(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def version_control_rolling_file(self, path, curr_version, max_version):
        new_version = curr_version + 1
        if new_version > max_version: return

        old_path = "{}.{}".format(path, curr_version) if curr_version!=0 else path
        new_path = "{}.{}".format(path, new_version) if new_version!=0 else path
        if os.path.exists(old_path):
            if os.path.exists(new_path):
                self.version_control_rolling_file(path, curr_version+1, max_version)
                os.remove(new_path)
            shutil.copyfile(old_path, new_path)

