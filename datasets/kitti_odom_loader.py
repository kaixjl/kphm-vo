# Mostly based on the code written by Tinghui Zhou: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data/kitti/kitti_odom_loader.py
from __future__ import division
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import os
from .odometry_dataset_loader import OdometryDatasetLoader, OdometryDatasetRelatedLoader, Subsequence, pil_loader

class KittiOdometryLoader(OdometryDatasetLoader):
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=5,
                 sequences: List[int] = list(range(11)),
                 remove_static = False,
                 camera_ids=[2,3]):
        '''
        '''
        self._init(dataset_dir,
                   img_height,
                   img_width,
                   seq_length,
                   sequences,
                   camera_ids if camera_ids is not None else [2,3])

        self.seqs_and_frames = self.seqs_and_frames # type: dict[int, List[int]]
        self.remove_static_list = {seq:None for seq in self.seqs} # type: List[Tuple[int]]
        self.remove_static = remove_static
        if self.remove_static:
            self._collect_static_frames()

        self._collect_frames()

    def _collect_static_frames(self):
        path_remove_static = os.path.join(os.path.dirname(__file__), "static_frames.txt")
        with open(path_remove_static, 'r') as f:
            remove_static_list_tmp = f.read().splitlines()
            remove_static_list_tmp = [tuple(i.split()) for i in remove_static_list_tmp]
            for i in remove_static_list_tmp:
                self.remove_static_list[int(i[0])] = tuple(int(idx) for idx in i[1:])

    def _load_sequences(self):
        '''
        return List of sequences
        '''
        seqs = [] # type: List[int]
        sequences_dir = os.path.join(self.dataset_dir, "sequences")
        sequences = os.listdir(sequences_dir)
        for seq in sequences:
            seq_path = os.path.join(sequences_dir, seq)
            if(os.path.isdir(seq_path)):
                seqs.append(int(seq))
        return seqs

    def _collect_frames_filter(self, seq, frame_id):
        # type: (Any, Any) -> bool
        '''
        return what self.seqs_and_frames store per sequence if frame_id in seq should be collected, else None.

        ## Parameters:
        - seq: is each of what _load_sequences returns.
        - frame_id: is the frame filename without extension.
        '''
        if self.remove_static and self.remove_static_list[seq] is not None and int(frame_id) in self.remove_static_list[seq]:
            return None
        else:
            return int(frame_id)

    def _seq_dir(self, seq):
        # type: (Any) -> str
        '''
        return dir of a sequence.
        '''
        return os.path.join(self.dataset_dir, 'sequences', '{:02d}'.format(seq))

    def _seq_img_dir(self, seq, camera_id=None):
        # type: (Any, Any) -> str
        '''
        return dir storing the frame images of a sequence.
        '''
        if camera_id is None: camera_id = self.camera_ids[0]
        assert(camera_id in self.camera_ids)
        seq_dir = self._seq_dir(seq)
        img_dir = os.path.join(seq_dir, 'image_{}'.format(camera_id))
        return img_dir

    def load_image_original(self, seq_id, frame_id, camera_id=None):
        img_file = os.path.join(self._seq_img_dir(seq_id, camera_id), ('{:0'+str(self._filename_len)+'d}.{}').format(frame_id, self.ext))
        img = pil_loader(img_file)
        # img = imageio.imread(img_file)
        return img

    def load_projection_matrix_original(self, seq_id, camera_id=None):
        # type: (int, int) -> np.ndarray
        '''
        return a ndarray with shape (3, 4)
        '''
        calib_file = os.path.join(self.dataset_dir, 'sequences', '{:02d}/calib.txt'.format(seq_id))
        proj_c2p, _ = self.read_calib_file(calib_file, camera_id if camera_id is not None else self.camera_ids[0])
        p = proj_c2p
        return p

    def load_intrinsics_original(self, seq_id, camera_id=None):
        # type: (int, int) -> np.ndarray
        '''
        return a ndarray with shape (3, 3)
        '''
        calib_file = os.path.join(self.dataset_dir, 'sequences', '{:02d}/calib.txt'.format(seq_id))
        proj_c2p, _ = self.read_calib_file(calib_file, camera_id)
        intrinsics = proj_c2p[:3, :3]
        return intrinsics

    def read_calib_file(self, filepath, camera_id=2):
        # type: (str, int) -> Tuple[np.ndarray, np.ndarray]
        """Read in a calibration file and parse into a dictionary."""
        if camera_id is None: camera_id = 2
        with open(filepath, 'r') as f:
            C = f.readlines()
        def parseLine(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32) # type: np.ndarray
            return data
        proj_c2p = parseLine(C[camera_id], shape=(3,4))
        proj_v2c = parseLine(C[-1], shape=(3,4))
        filler = np.array([0, 0, 0, 1]).reshape((1,4))
        proj_v2c = np.concatenate((proj_v2c, filler), axis=0) #type: np.ndarray
        return proj_c2p, proj_v2c

    def _load_pose_gt_abs_matrix(self, seq_id):
        '''
        return a ndarray with shape (N, 4, 4)
        '''
        matrix12 = np.reshape(np.loadtxt(os.path.join(self.dataset_dir, "poses", "{:02d}.txt".format(seq_id))), (-1, 3, 4)).astype(np.float32)
        matrix4 = np.zeros((matrix12.shape[0], 1, 4), dtype=np.float32)
        matrix4[:,:,3] = 1
        matrix16 = np.concatenate((matrix12, matrix4), axis=1)
        return matrix16

    def get_pose_gt_abs_matrix_all_except_static(self, seq_id) -> np.ndarray:
        '''
        If remove_static is False when create the object, this method will return all gt like get_pose_gt_abs_matrix_all method.

        ## Return

        Refer to frame_id in parameters
        '''
        pose_gt = self.get_pose_gt_abs_matrix_all(seq_id)

        if self.remove_static_list[seq_id] is None:
            return pose_gt
        else:
            non_static = tuple(i for i in range(pose_gt.shape[0]) if i not in self.remove_static_list[seq_id])
            return pose_gt[non_static,:,:]

class KittiOdometryRelatedLoader(OdometryDatasetRelatedLoader):
    def __init__(self,
                 dataset_related_dir,
                 subdir,
                 img_height=128,
                 img_width=416,
                 camera_ids=[2,3]):
        '''
        dataset_related directory loader

        ## Parameter:

        - subdir: subdirectory of dataset_related_dir. should have the same structure as the sequences folder in original dataset.
        '''
        super().__init__(dataset_related_dir,
                         subdir,
                         img_height,
                         img_width,
                         camera_ids if camera_ids is not None else [2,3])

        self._filename_len = 10

    def _seq_dir(self, seq):
        # type: (Any) -> str
        '''
        return dir of a sequence.
        '''
        return os.path.join(self.dataset_related_dir, self.subdir, '{:02d}'.format(seq))
    
    def _seq_img_dir(self, seq, camera_id=None):
        # type: (Any, Any) -> str
        '''
        return dir storing the frame images of a sequence.
        '''
        if camera_id is None: camera_id = self.camera_ids[0]
        assert(camera_id in self.camera_ids)
        seq_dir = self._seq_dir(seq)
        img_dir = os.path.join(seq_dir, 'image_{}'.format(camera_id))
        return img_dir

    def load_image_original(self, seq_id, frame_id, camera_id=None):
        img_file = os.path.join(self._seq_img_dir(seq_id, camera_id), ('{:0'+str(self._filename_len)+'d}.{}').format(frame_id, 'png'))
        # img = imageio.imread(img_file)
        img = pil_loader(img_file)
        return img
