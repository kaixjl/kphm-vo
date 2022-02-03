from typing import Callable, Dict, Tuple, Iterable, List, Union
import os
import os.path
import sys
import time
import random
import numpy as np
import torch
import torch.distributed as dist
import argparse
from models.misc import create_networks
from utils import Timer
from params import get_param
from trainer import Train

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--epoch_size", type=int, help="epoch size")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--not_cuda", help="use cpu but cuda.", action='store_true')
    parser.add_argument("--config_name", "-c", type=str, help="config name. used to specify the tag of this running")
    parser.add_argument("--split_tag", type=str, help="split tag")
    parser.add_argument("--dataset_related_dir", type=str, help="dataset_related_dir")
    parser.add_argument("--dataset_dir", type=str, help="dataset_dir")
    parser.add_argument("--comment", type=str, help="comment about this training")
    parser.add_argument("--cudas", nargs="+", type=int, help="specify cudas to be used.")
    parser.add_argument("--use_config", "-u", nargs="+", type=str, help="specify which config to be used.")
    parser.add_argument("--debug", nargs="?", const=True, type=int, help="debug mode.")
    parser.add_argument("--cudnn_backend", nargs="?", const=True, type=int, help="if use cudnn backend. True if this flag is not be used or value of this flag is not specified.")
    parser.add_argument("--multi_gpus", nargs="?", const=-1, type=int, help="Number of gpus to be used. 0 means don't use DistributedDataParallel. -1 means use all gpus visible.")
    parser.add_argument("--network", type=int, help="specify network to be used.")

    args0 = parser.parse_args()

    # param.cuda_visible_devices = args0.cuda_visible_devices
    return args0

def create_param(args0):
    param = get_param(args0.use_config)

    def choose(a, b):
        return a if a is not None else b

    param.config_name = choose(args0.config_name, param.config_name)
    param.epoch_size = choose(args0.epoch_size, param.epoch_size)
    param.batch_size = choose(args0.batch_size, param.batch_size)
    param.lr = choose(args0.lr, param.lr)
    param.split_tag = choose(args0.split_tag, param.split_tag)
    param.dataset_related_dir = choose(args0.dataset_related_dir, param.dataset_related_dir)
    param.dataset_dir = choose(args0.dataset_dir, param.dataset_dir)
    param.not_cuda = args0.not_cuda
    param.resume = args0.resume
    param.comment = args0.comment
    param.cuda_visible_devices = choose(args0.cudas, param.cuda_visible_devices)
    param.debug = choose(bool(args0.debug) if args0.debug is not None else None, param.debug)
    param.cudnn_backend = choose(bool(args0.cudnn_backend) if args0.cudnn_backend is not None else None, param.cudnn_backend)
    param.is_distributed = choose(args0.multi_gpus, param.is_distributed)
    if param.is_distributed == -1: param.is_distributed = len(param.cuda_visible_devices)
    param.network = choose(args0.network, param.network)

    return param
    

def main():
    args0 = parse_arg()
    param = create_param(args0)
    if param.is_distributed == 0:
        run(0, 0, False, args0)
    else:
        torch.multiprocessing.spawn(run, (param.is_distributed, True, args0, int(time.time())), param.is_distributed)


def run(rank, world_size, is_distributed, args0, seed=None):
    param = create_param(args0)
    if is_distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '51351'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    if seed is not None:
        set_seed(seed)  # 保证各个进程的模型初始化时参数保持一致
        

    torch.backends.cudnn.enabled = param.cudnn_backend

    if not param.debug:
        result_dir = os.path.join("result", param.config_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        output_redirection = open(os.path.join(result_dir, "nohup{}.out".format(rank if rank!=0 else "")), 'w')
        stdout = sys.stdout
        sys.stdout = output_redirection

    nets = create_networks(param.network, param, True, param.img_height, param.img_width, True)
    if not isinstance(nets, Iterable):
        nets = (nets,)
    trainers = [] # type: List[Train]
    for net in nets:
        trainer_class = None
        trainer_class = Train
        trainer = trainer_class(param)
        trainer.set_networks(net)
        trainers.append(trainer)

    for trainer in trainers:
        trainer.train_init()

    for trainer in trainers:
        trainer.train_start()

    with Timer() as t:
        for trainer in trainers:
            trainer.train_step(param.epoch_size, t)

    for trainer in trainers:
        trainer.train_end()

    if not param.debug:
        sys.stdout = stdout
    
    if is_distributed:
        dist.destroy_process_group()
    pass

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__=="__main__":
    main()
