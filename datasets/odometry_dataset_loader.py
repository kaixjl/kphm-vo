from __future__ import division
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import os
import imageio
import skimage.transform
from PIL import Image
import cv2 as cv
from utils.nputils import reverse_Rt_matrix, Rt_to_euler_trans_multi, pose_abs_to_first
import abc

def pil_loader(path):
    # type: (str) -> Image.Image
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.copy()

class OdometryDatasetLoader(abc.ABC):
    def __init__(self,
                 dataset_dir,
                 img_height=128,
                 img_width=416,
                 seq_length=5,
                 sequences: List = None,
                 camera_ids: List = None):
        '''
        '''
        self._init(dataset_dir,
                   img_height,
                   img_width,
                   seq_length,
                   sequences,
                   camera_ids)

        self._collect_frames()
    
    def _init(self,
              dataset_dir,
              img_height=128,
              img_width=416,
              seq_length=5,
              sequences: List = None,
              camera_ids: List = None):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.seqs = sequences if sequences is not None else self._load_sequences()
        self.pose_gt = {seq:None for seq in self.seqs}
        self.seqs_and_frames = {seq:None for seq in self.seqs} # type: dict[List]
        self.camera_ids = camera_ids if camera_ids is not None else [0]

    @abc.abstractmethod
    def _load_sequences(self):
        '''
        return List of sequences
        '''
        raise NotImplementedError()

    @property
    def sequences(self):
        return self.seqs

    def frames_in_sequence(self, seq_id):
        return self.seqs_and_frames[seq_id]

    @property
    def zoom_x(self):
        return self._zoom_x

    @property
    def zoom_y(self):
        return self._zoom_y

    @abc.abstractmethod
    def _collect_frames_filter(self, seq, frame_id):
        # type: (Any, Any) -> bool
        '''
        return what self.seqs_and_frames store per sequence if frame_id in seq should be collected, else None.

        ## Parameters:
        - seq: is each of what _load_sequences returns.
        - frame_id: is the frame filename without extension.
        '''
        return frame_id

    @ abc.abstractmethod
    def _seq_dir(self, seq):
        # type: (Any) -> str
        '''
        return dir of a sequence.
        '''
        raise NotImplementedError()

    @ abc.abstractmethod
    def _seq_img_dir(self, seq, camera_id=None):
        # type: (Any, Any) -> str
        '''
        return dir storing the frame images of a sequence.
        '''
        raise NotImplementedError()

    def _collect_frames(self):
        # type: () -> None
        '''
        collect frames to Lists, save frame file ext, and check dataset.
        save frames per seq in self.seqs_and_frames[seq]
        '''
        self.ext = None
        for seq in self.seqs:
            # print("Checking sequence {}".format(seq))
            frames_in_seq = []
            img_dir = self._seq_img_dir(seq)
            frames = os.listdir(img_dir)
            frames.sort()
            for frame_filename in frames:
                if self.ext == None:
                    self.ext = frame_filename[-3:] # save frame ext
                    img = imageio.imread(os.path.join(img_dir, frame_filename))
                    self.ori_height = img.shape[0]
                    self.ori_width = img.shape[1]
                    self.cut_u, self.cut_v, self.cut_h, self.cut_w = 0, 0, self.ori_height, self.ori_width
                    self._zoom_y = self.img_height/self.cut_h
                    self._zoom_x = self.img_width/self.cut_w
                    self._filename_len = len(frame_filename[:-4])
                frames_in_seq_item = self._collect_frames_filter(seq, frame_filename[:-4])
                if frames_in_seq_item is not None:
                    frames_in_seq.append(frames_in_seq_item)
                    # assert(("{:0"+str(self._filename_len)+"d}.{}").format(i, self.ext)==frame_filename)
            self.seqs_and_frames[seq] = frames_in_seq
        # self.num_train = len(self.train_frames)

    def _is_valid_sample(self, seq_id, tgt_idx, seq_length):
        N = len(self.seqs_and_frames[seq_id])
        half_offset = (seq_length - 1)//2
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        return True

    def _generate_subsequence_list(self, seq_id, tgt_idx, seq_length):
        if not self._is_valid_sample(seq_id, tgt_idx, seq_length):
            return None
        half_offset = int((seq_length - 1)/2)
        subseq_frame_ids = []
        for o in range(-half_offset, half_offset+1):
            curr_frame_id = self.seqs_and_frames[seq_id][tgt_idx + o]
            subseq_frame_ids.append(curr_frame_id)
        return subseq_frame_ids

    def load_subsequence(self, seq_id, tgt_idx):
        subseq_frame_ids = self._generate_subsequence_list(seq_id, tgt_idx, self.seq_length)

        if subseq_frame_ids == None:
            return None
            
        intrinsics = self.load_intrinsics(seq_id)     
        subsequence = Subsequence(intrinsics, seq_id, subseq_frame_ids, self.seqs_and_frames[seq_id][tgt_idx])

        return subsequence

    def yield_subsequences(self, seq_id, interval = None, start = 0, end = None):
        '''
        generate subsequences with target frame id from *start* to *end* by *interval*.

        ## Parameters

        - seq_id: sequence id. 

        - interval: sample interval. Same as seq_length if None.

        - start: start frame id. Default is 0. If *start* is less than seq_length//2, *start* will be set to seq_length//2.
        
        - end: end frame idx. len(frames_in_sequence(seq_id)) if None. Subsequences target frame id will less than end frame id (excluded).
        '''
        if end == None:
            end = len(self.seqs_and_frames[seq_id])
        if interval == None:
            interval = self.seq_length

        curr_frame_idx = start
        if start < self.seq_length//2:
            curr_frame_idx = self.seq_length//2

        while(curr_frame_idx < end):
            subseq = self.load_subsequence(seq_id, curr_frame_idx)
            if subseq != None:
                yield subseq
            curr_frame_idx += interval

    @abc.abstractmethod
    def load_image_original(self, seq_id, frame_id, camera_id=None):
        img_file = os.path.join(self._seq_img_dir(seq_id, camera_id), ('{}.{}').format(frame_id, self.ext))
        img = pil_loader(img_file)
        # img = imageio.imread(img_file)
        return img

    def load_image(self, seq_id, frame_id, camera_id=None):
        curr_img = self.load_image_original(seq_id, frame_id, camera_id)
        cut_u, cut_v, cut_h, cut_w = self.cut_u, self.cut_v, self.cut_h, self.cut_w 
        curr_img = curr_img.crop((cut_v, cut_u, cut_v+cut_w, cut_u+cut_h)).resize((self.img_width, self.img_height), Image.ANTIALIAS)
        # curr_img = skimage.transform.resize(curr_img[cut_u:cut_u+cut_h, cut_v:cut_v+cut_w], (self.img_height, self.img_width))
        
        return curr_img

    @abc.abstractmethod
    def load_projection_matrix_original(self, seq_id, camera_id=None):
        # type: (Any, int) -> np.ndarray
        '''
        return a ndarray with shape (3, 4)
        '''
        raise NotImplementedError()
    
    def load_projection_matrix(self, seq_id, camera_id=None):
        # type: (Any, int, Optional[Tuple[int, int, int, int]]) -> np.ndarray
        p = self.load_projection_matrix_original(seq_id, camera_id)
        cut_u, cut_v, cut_h, cut_w = self.cut_u, self.cut_v, self.cut_h, self.cut_w 
        zoom_x, zoom_y = self.zoom_x, self.zoom_y
        p = self.scale_projection_matrix(p, zoom_x, zoom_y, cut_v, cut_u)
        return p

    @abc.abstractmethod
    def load_intrinsics_original(self, seq_id, camera_id=None):
        # type: (int, int) -> np.ndarray
        '''
        return a ndarray with shape (3, 3)
        '''
        raise NotImplementedError()

    def load_intrinsics(self, seq_id, camera_id=None):
        # type: (int, int, Optional[Tuple[int, int, int, int]]) -> np.ndarray
        intrinsics = self.load_intrinsics_original(seq_id, camera_id)
        cut_u, cut_v, cut_h, cut_w = self.cut_u, self.cut_v, self.cut_h, self.cut_w 
        zoom_x, zoom_y = self.zoom_x, self.zoom_y
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y, cut_v, cut_u)
        return intrinsics

    def scale_intrinsics(self,mat, sx, sy, dx = 0, dy = 0):
        out = np.copy(mat)
        out[0,0] = out[0,0] * sx
        out[0,2] = (out[0,2] - dx) * sx
        out[1,1] = out[1,1] * sy
        out[1,2] = (out[1,2] - dy) * sy
        return out

    def scale_projection_matrix(self,mat, sx, sy, dx = 0, dy = 0):
        out = np.copy(mat)
        out[0,0] = out[0,0] * sx
        out[0,2] = (out[0,2] - dx) * sx
        out[0,3] = out[0,3] * sx
        out[1,1] = out[1,1] * sy
        out[1,2] = (out[1,2] - dy) * sy
        out[1,3] = out[1,3] * sy
        return out

    @abc.abstractclassmethod
    def _load_pose_gt_abs_matrix(self, seq_id):
        '''
        return a ndarray with shape (N, 4, 4)
        '''
        raise NotImplementedError()
    
    def get_pose_gt_abs_matrix_all(self, seq_id) -> np.ndarray:
        '''
        ## Return

        Refer to frame_id in parameters
        '''
        if seq_id not in self.pose_gt.keys() or self.pose_gt[seq_id] is None:
            self.pose_gt[seq_id] = self._load_pose_gt_abs_matrix(seq_id)

        return self.pose_gt[seq_id]

    def get_pose_gt_abs_matrix(self, seq_id, frame_id: Union[int, Tuple[int]]) -> np.ndarray:
        '''
        ## Parameters

        - frame_id: int or tuple is acceptable. 
            + If an int is provided, an ndarray with shape (4, 4) is returned. 
            + If a tuple is provided, an ndarray with shape (N, 4, 4) is returned, with N original pose groundtruth. 
              If you need pose related to frame_id[0], you need some post-process by yourself.

        ## Return

        Refer to frame_id in parameters
        '''
        if seq_id not in self.pose_gt.keys() or self.pose_gt[seq_id] is None:
            self.pose_gt[seq_id] = self._load_pose_gt_abs_matrix(seq_id)

        if type(frame_id) is int:
            return self.pose_gt[seq_id][frame_id]
        elif type(frame_id) is tuple:
            return np.stack([self.pose_gt[seq_id][i] for i in frame_id])

    def get_pose_gt_abs_euler_trans(self, seq_id, frame_id):
        return Rt_to_euler_trans_multi(self.get_pose_gt_abs_matrix(seq_id, frame_id))

    def get_pose_gt_abs_matrix_to_first(self, seq_id, frame_id: Union[int, Tuple[int]]) -> np.ndarray:
        '''
        ## Parameters

        - frame_id: int or tuple is acceptable. 
            + If an int is provided, an ndarray with shape (4, 4) is returned. 
            + If a tuple is provided, an ndarray with shape (N, 4, 4) is returned, with N original pose groundtruth. 
              If you need pose related to frame_id[0], you need some post-process by yourself.

        ## Return

        Refer to frame_id in parameters
        '''
        if seq_id not in self.pose_gt.keys() or self.pose_gt[seq_id] is None:
            self.pose_gt[seq_id] = self._load_pose_gt_abs_matrix(seq_id)

        if type(frame_id) is int:
            return self.pose_gt[seq_id][frame_id]
        elif type(frame_id) is tuple:
            return pose_abs_to_first(np.stack([self.pose_gt[seq_id][i] for i in frame_id]))

    def get_pose_gt_abs_euler_trans_to_first(self, seq_id, frame_id):
        return Rt_to_euler_trans_multi(self.get_pose_gt_abs_matrix_to_first(seq_id, frame_id))

    def get_pose_gt_rel_matrix(self, seq_id, frame_id: Union[int, Tuple[int]]) -> np.ndarray:
        '''
        ## Parameters

        - frame_id: int or tuple is acceptable. 
            + If an int is provided, an ndarray with shape (4, 4) is returned, which is pose related to former one frame. 
            + If a tuple is provided, an ndarray with shape (N-1, 4, 4) is returned, with latter N-1 relative pose related to former N-1 frames in frame_id. In this case frame_id must be sorted increasingly.

        ## Return

        Refer to frame_id in parameters
        '''
        if seq_id not in self.pose_gt.keys() or self.pose_gt[seq_id] is None:
            self.pose_gt[seq_id] = self._load_pose_gt_abs_matrix(seq_id)

        # if self.pose_gt_rel[seq_id] is None:
        #     self.pose_gt_rel[seq_id] = self._load_pose_gt_rel_matrix(seq_id)
            
        if type(frame_id) is int:
            # return self.pose_gt_rel[seq_id][frame_id]
            if frame_id==0:
                return np.eye(4)
            else:
                return np.matmul(reverse_Rt_matrix(self.pose_gt[seq_id][frame_id - 1]), self.pose_gt[seq_id][frame_id])
        elif type(frame_id) is tuple:
            assert(all(i<j for i,j in zip(frame_id[:-1], frame_id[1:])))
            # return np.stack([reduce(lambda acc, y: np.matmul(acc,y), self.pose_gt_rel[seq_id][i+1:j+1]) for i,j, in zip(frame_id[:-1], frame_id[1:])])
            return np.stack([np.matmul(reverse_Rt_matrix(self.pose_gt[seq_id][i]), self.pose_gt[seq_id][j]) for i,j, in zip(frame_id[:-1], frame_id[1:])])



    def get_pose_gt_rel_euler_trans(self, seq_id, frame_id):
        return Rt_to_euler_trans_multi(self.get_pose_gt_rel_matrix(seq_id, frame_id))

class OdometryDatasetLoaderUndistorted(OdometryDatasetLoader, metaclass=abc.ABCMeta):

    def load_image_original_undistort(self, seq_id, frame_id, camera_id=None):
        img_ori = np.array(self.load_image_original(seq_id, frame_id, camera_id))
        intrinsics = self.load_intrinsics_original(seq_id, camera_id)
        distCoeffs = self.load_distortion_coefficients(seq_id, camera_id)
        img_ori_rec = cv.undistort(img_ori, intrinsics, distCoeffs)

        return img_ori_rec

    def load_image_undistort(self, seq_id, frame_id, camera_id=None):
        curr_img = self.load_image_original_undistort(seq_id, frame_id, camera_id)
        cut_u, cut_v, cut_h, cut_w = self.cut_u, self.cut_v, self.cut_h, self.cut_w 
        zoom_x, zoom_y = self.zoom_x, self.zoom_y
        # curr_img = curr_img.crop((cut_v, cut_u, cut_v+cut_w, cut_u+cut_h)).resize((self.img_width, self.img_height), Image.ANTIALIAS)
        curr_img = skimage.img_as_ubyte(skimage.transform.resize(curr_img[cut_u:cut_u+cut_h, cut_v:cut_v+cut_w], (self.img_height, self.img_width)))
        
        return curr_img

    @abc.abstractmethod
    def load_distortion_coefficients(self, seq_id, camera_id=None):
        raise NotImplementedError()

class Subsequence:
    def __init__(self, intrinsics, seq_id, subseq_frames_id, tgt_frame_id = None):
        self._intrinsics = intrinsics
        self._seq_id = seq_id
        self._subseq_frames_id = subseq_frames_id
        self._tgt_frame_id = tgt_frame_id if tgt_frame_id != None else subseq_frames_id[len(subseq_frames_id)//2]
        self._subseq_frames_img = None

    @property
    def intrinsics(self) -> np.ndarray:
        return self._intrinsics

    @property
    def seq_id(self) -> int:
        return self._seq_id

    @property
    def subseq_frames_id(self) -> List[int]:
        return self._subseq_frames_id

    @property
    def tgt_frame_id(self) -> int:
        return self._tgt_frame_id

    @property
    def subseq_frames_img(self) -> List[imageio.core.Image]:
        return self._subseq_frames_img

    def load_subseq_frames_img(self, odometry_dataset_loader: OdometryDatasetLoader, camera_id=None):
        if self._subseq_frames_img != None:
            return
        
        self._subseq_frames_img = []
        for frame_id in self._subseq_frames_id:
            self._subseq_frames_img.append(odometry_dataset_loader.load_image(self._seq_id, frame_id, camera_id=None))

class OdometryDatasetRelatedLoader(object):
    def __init__(self,
                 dataset_related_dir,
                 subdir,
                 img_height=128,
                 img_width=416,
                 camera_ids:List=None):
        '''
        dataset_related directory loader

        ## Parameter:

        - subdir: subdirectory of dataset_related_dir. should have the same structure as the sequences folder in original dataset.
        '''
        self.dataset_related_dir = dataset_related_dir
        self.img_height = img_height
        self.img_width = img_width
        self.subdir = subdir
        self.camera_ids = camera_ids if camera_ids is not None else [0]

    @abc.abstractmethod
    def _seq_dir(self, seq):
        # type: (Any) -> str
        '''
        return dir of a sequence.
        '''
        raise NotImplementedError()
    
    @abc.abstractmethod
    def _seq_img_dir(self, seq, camera_id=None):
        # type: (Any, Any) -> str
        '''
        return dir storing the frame images of a sequence.
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def load_image_original(self, seq_id, frame_id, camera_id=None):
        img_file = os.path.join(self._seq_img_dir(seq_id, camera_id), ('{}.{}').format(frame_id, 'png'))
        # img = imageio.imread(img_file)
        img = pil_loader(img_file)
        return img

    def load_image(self, seq_id, frame_id, camera_id=None):
        curr_img = self.load_image_original(seq_id, frame_id, camera_id)
        curr_img = curr_img.resize((self.img_width, self.img_height), Image.ANTIALIAS)
        return curr_img
