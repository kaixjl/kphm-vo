# Based on DeepVO-pytorch
from .kitti_odom_loader import KittiOdometryLoader, KittiOdometryRelatedLoader
# import os
# import glob
from typing import Any, List, Tuple, Union
import torch
from torchvision import transforms
from .odometry_dataset import OdometryDatasetSequenceDataset, OdometryDataset, OdometryDatasetGTEnum

class KittiOdometrySequenceDataset(OdometryDatasetSequenceDataset):
    def __init__(self,
                 dataset: Union[str, List[Tuple[int, ...]]],
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 return_intrinsics = False,
                 gt_type: OdometryDatasetGTEnum = OdometryDatasetGTEnum.NONE,
                 gt_rel_path = None,
                 aug=False,):
        # type: (Union[str, List[Tuple[str, ...]]], str, int, int, bool, OdometryDatasetGTEnum, Any, bool) -> None
        '''
        ## Parameters:
        - dataset: path to split file or List of Tuple(seq_id, frame1, frame2, ..., frameN)
        '''

        if type(dataset) is str:
            with open(dataset, 'r') as f:
                dataset = f.read() # type: str
            dataset = self.parse_dataset_text(dataset)

        seqs_id = list(set(row[0] for row in dataset)) # type: List[int]
        loader = KittiOdometryLoader(dataset_dir, img_height, img_width, sequences=seqs_id)
 
        super().__init__(dataset, loader, return_intrinsics, gt_type, gt_rel_path, aug)

    @staticmethod
    def parse_dataset_text(dataset_text: str) -> List[Tuple[int, ...]]:
        # type: (str) -> List[Tuple[int, ...]]
        '''
        parse dataset text to a List of Tuple of int.

        ## Parameters

        - dataset_text: Each row is whitespace seperated string, as a subsequence sample. The first item of each row is seq_id, and the rest are frame ids in the subsequence.
        For example, '3 0 1 2 3' means a subsequence in seq_id=3 with 4 frames, frame_id=0, 1, 2, 3.
        '''
        dataset_text = dataset_text.splitlines()
        dataset_text = [tuple(int(i) for i in line.split()) for line in dataset_text]
        return dataset_text

class KittiOdometryDataset(OdometryDataset):
    def __init__(self,
                 seq_id,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 return_intrinsics = False,
                 gt_type: OdometryDatasetGTEnum = OdometryDatasetGTEnum.NONE,
                 gt_rel_path = None):
        # type: (int, str, int, int, bool, OdometryDatasetGTEnum, Any, bool) -> None
        '''
        ## Parameters:
        - odometry_dataset_loader: with len(seqs) == 1
        '''

        loader = KittiOdometryLoader(dataset_dir, img_height, img_width, sequences=[seq_id])

        super().__init__(loader, return_intrinsics, gt_type, gt_rel_path)

class KittiOdometrySequenceWithHeatmapDataset(KittiOdometrySequenceDataset):
    def __init__(self, dataset: Union[str, List[Tuple[int, ...]]], dataset_dir, dataset_related_dir, img_height=128, img_width=416, return_intrinsics = False, gt_type: OdometryDatasetGTEnum = OdometryDatasetGTEnum.NONE, gt_rel_path = None, return_heatmap = False, aug=False):
        super().__init__(dataset, dataset_dir, img_height, img_width, return_intrinsics, gt_type, gt_rel_path, aug)

        self.return_heatmap = return_heatmap
        self.heatmap_loader = None

        if return_heatmap:
            self.heatmap_loader = KittiOdometryRelatedLoader(dataset_related_dir, 'kpheatmap', img_height, img_width)

    def __getitem__(self, index):
        rst = super().__getitem__(index)

        row = self.dataset[index // 2 if self.aug else index]
        seq_id = row[0]
        frame_ids = row[1:]

        if self.return_heatmap:

            transform = transforms.ToTensor()

            frames = []
            for frame_id in frame_ids:
                frames.append(transform(self.heatmap_loader.load_image(seq_id, frame_id)).to(dtype=torch.float32))
            frames = torch.stack(frames) # shape (t, c, h, w)
            rst += (frames,)

        return rst
