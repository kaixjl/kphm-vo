# Based on DeepVO-pytorch
from abc import abstractclassmethod, abstractstaticmethod
import abc
from enum import Enum
from .odometry_dataset_loader import OdometryDatasetLoader
from typing import Any, List, Tuple, Union
import random
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class OdometryDatasetGTEnum(Enum):
    NONE = 0
    ABSOLUTE = 1 # Not used
    RELATIVE = 2

class OdometryDatasetSequenceDataset(Dataset, metaclass = abc.ABCMeta):
    def __init__(self,
                 dataset,
                 odometry_dataset_loader,
                 return_intrinsics = False,
                 gt_type: OdometryDatasetGTEnum = OdometryDatasetGTEnum.NONE,
                 gt_rel_path = None,
                 aug_color=False):
        # type: (List[Tuple], OdometryDatasetLoader, bool, OdometryDatasetGTEnum, Any, bool) -> None
        '''
        ## Parameters:
        - dataset: List of Tuple(seq_id, frame1, frame2, ..., frameN)
        '''

        self.dataset = dataset # type: List[Tuple[int, ...]]
        self.loader = odometry_dataset_loader
        self.gt_type = gt_type if gt_type is not None else OdometryDatasetGTEnum.NONE
        self.gt_rel = None # TODO: load gt_rel_path file
        self.return_intrinsics = return_intrinsics

        self.aug_color = aug_color
        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.aug = self.aug_color
        
        self._seq_len = len(self.dataset[0]) - 1

    def __getitem__(self, index):
        row = self.dataset[index // 2 if self.aug else index]
        seq_id = row[0]
        frame_ids = row[1:]

        transform = transforms.ToTensor()
        if self.aug and self.aug_color and index % 2 == 1:
            self._color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            self._color_aug = (lambda x: x)

        frames = []
        for frame_id in frame_ids:
            frame = self.loader.load_image(seq_id, frame_id)
            frame = self._color_aug(frame)
            frame = transform(frame)
            frame = frame.to(dtype=torch.float32)
            frames.append(frame)
        frames = torch.stack(frames) # shape (t, c, h, w)
        rst = (frames,)

        if self.return_intrinsics:
            intrinsics = torch.tensor(self.loader.load_intrinsics(seq_id), dtype=torch.float32)
            rst += (intrinsics,)

        if self.gt_type == OdometryDatasetGTEnum.ABSOLUTE:
            gt = self.loader.get_pose_gt_abs_matrix_to_first(seq_id, frame_ids).to(dtype=torch.float32)
            rst += (gt,)
        elif self.gt_type == OdometryDatasetGTEnum.RELATIVE:
            gt = self.loader.get_pose_gt_rel_matrix(seq_id, frame_ids) # NOTE: assume frame_ids are increasing.
            gt = torch.tensor(gt)
            rst += (gt,)

        return rst # return frames[, intrinsics][, gt]


    def __len__(self):
        return len(self.dataset)*2 if self.aug else len(self.dataset)

    @property
    def seq_len(self):
        return self._seq_len

class OdometryDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self,
                 odometry_dataset_loader,
                 return_intrinsics = False,
                 gt_type: OdometryDatasetGTEnum = OdometryDatasetGTEnum.NONE,
                 gt_rel_path = None):
        # type: (OdometryDatasetLoader, bool, OdometryDatasetGTEnum, Any) -> None
        '''
        ## Parameters:
        - odometry_dataset_loader: with len(seqs) == 1
        '''
        assert(len(odometry_dataset_loader.seqs) == 1)
        self.seq_id = odometry_dataset_loader.seqs[0]
        self.loader = odometry_dataset_loader
        self.gt_type = gt_type if gt_type is not None else OdometryDatasetGTEnum.NONE
        self.gt_rel = None # TODO: load gt_rel_path file
        self.return_intrinsics = return_intrinsics
                

    def __getitem__(self, index):
        seq_id = self.seq_id
        frames = self.loader.frames_in_sequence(seq_id)
        frame_id = frames[index]

        transform = transforms.ToTensor()

        frame = transform(self.loader.load_image(seq_id, frame_id)).to(dtype=torch.float32)
        rst = (frame,)

        if self.return_intrinsics:
            intrinsics = torch.tensor(self.loader.load_intrinsics(seq_id), dtype=torch.float32)
            rst += (intrinsics,)

        if self.gt_type == OdometryDatasetGTEnum.ABSOLUTE:
            gt = self.loader.get_pose_gt_abs_matrix(seq_id, frame_id) # TODO: add support to euler-trans / angle-axis / 4
            gt = torch.tensor(gt, dtype=torch.float32)
            rst += (gt,)
        elif self.gt_type == OdometryDatasetGTEnum.RELATIVE:
            gt = self.loader.get_pose_gt_rel_matrix(seq_id, frame_id) # TODO: add support to euler-trans / angle-axis / 4
            gt = torch.tensor(gt, dtype=torch.float32)
            rst += (gt,)

        return rst

    def __len__(self):
        return len(self.loader.frames_in_sequence(self.seq_id))
