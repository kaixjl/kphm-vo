# Based on DeepVO-pytorch
from typing import Iterable, Optional, OrderedDict
from itertools import accumulate
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

# epsilon for testing whether a number is close to zero
_EPS = torch.finfo(float).eps * 4.0
PI = 3.14159265358979

def is_rotation_matrix(R: torch.Tensor) :
    Rt = torch.transpose(R)
    shouldBeIdentity = torch.matmul(Rt, R)
    I = torch.eye(3, dtype = R.dtype, device=R.device)
    n = torch.norm(I - shouldBeIdentity)
    return n < 1e-6
    
def R_from_euler(theta: torch.Tensor) :
    R_x = torch.tensor([[1,                     0,                      0                   ],
                        [0,                     torch.cos(theta[0]),    -torch.sin(theta[0])],
                        [0,                     torch.sin(theta[0]),    torch.cos(theta[0]) ]
                        ])
    R_y = torch.tensor([[torch.cos(theta[1]),   0,                      torch.sin(theta[1]) ],
                        [0,                     1,                      0                   ],
                        [-torch.sin(theta[1]),  0,                      torch.cos(theta[1]) ]
                        ])
    R_z = torch.tensor([[torch.cos(theta[2]),   -torch.sin(theta[2]),   0                   ],
                        [torch.sin(theta[2]),   torch.cos(theta[2]),    0                   ],
                        [0,                     0,                      1                   ]
                        ])
    R = torch.matmul(R_x, torch.matmul( R_y, R_z ))
    return R

# def R_to_euler(matrix: torch.Tensor):
    
#     # y-x-z Tait–Bryan angles intrincic
#     # the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py
    
#     i = 2
#     j = 0
#     k = 1
#     repetition = 0
#     frame = 1
#     parity = 0
    

#     M = torch.tensor(matrix, dtype=torch.float64, device=matrix.device)[:3, :3]
#     if repetition:
#         sy = torch.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
#         if sy > _EPS:
#             ax = torch.atan2( M[i, j],  M[i, k])
#             ay = torch.atan2( sy,       M[i, i])
#             az = torch.atan2( M[j, i], -M[k, i])
#         else:
#             ax = torch.atan2(-M[j, k],  M[j, j])
#             ay = torch.atan2( sy,       M[i, i])
#             az = 0.0
#     else:
#         cy = torch.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
#         if cy > _EPS:
#             ax = torch.atan2( M[k, j],  M[k, k])
#             ay = torch.atan2(-M[k, i],  cy)
#             az = torch.atan2( M[j, i],  M[i, i])
#         else:
#             ax = torch.atan2(-M[j, k],  M[j, j])
#             ay = torch.atan2(-M[k, i],  cy)
#             az = 0.0

#     if parity:
#         ax, ay, az = -ax, -ay, -az
#     if frame:
#         ax, az = az, ax
#     return torch.tensor((ax, ay, az), dtype=matrix.dtype)

def R_to_euler(matrix):
    # type: (torch.Tensor) ->  torch.Tensor
    '''
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3) Z->Y->X，i.e. Mp=XYZp
    Returns
    -------
    np.ndarray([x, y, z])
     Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
     z = atan2(-r12, r11)
     y = asin(r13)
     x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = torch.tensor(matrix)
    cy_thresh = 1e-6
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flatten()
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = torch.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = torch.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
        x = torch.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = torch.atan2(r21,  r22)
        y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return torch.tensor([x, y, z])

def Rt_from_euler(rot, normalize_factor = None):
    # type: (torch.Tensor, float) -> torch.Tensor
    """
    ## Parameters:

    - rot # (b, 3)
    - normalize_factor: If not None, each of euler angle will be calculated tanh and multiplied with normalize_factor. If None, magnitude of vec will be respected as angle.

    ## Return:

    (b, 4, 4)
    """
    rot_0 = rot[:, 0:1].unsqueeze(1)
    rot_1 = rot[:, 1:2].unsqueeze(1)
    rot_2 = rot[:, 2:].unsqueeze(1)
    if normalize_factor is not None:
        rot_0 = torch.tanh(rot_0) * normalize_factor
        rot_1 = torch.tanh(rot_1) * normalize_factor
        rot_2 = torch.tanh(rot_2) * normalize_factor

    b, _ = rot.shape

    m111 = torch.tensor([[1, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 1]], dtype=torch.float, device=rot.device).expand((b, 4, 4))
    m122 = torch.tensor([[0, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 1]], dtype=torch.float, device=rot.device).expand((b, 4, 4))
    m133 = torch.tensor([[0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 1]], dtype=torch.float, device=rot.device).expand((b, 4, 4))
    mr3c = torch.tensor([[1, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0]], dtype=torch.float, device=rot.device).expand((b, 4, 4))
    mr2c = torch.tensor([[1, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 0]], dtype=torch.float, device=rot.device).expand((b, 4, 4))
    mr1c = torch.tensor([[0, 0, 0, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0], 
                         [0, 0, 0, 0]], dtype=torch.float, device=rot.device).expand((b, 4, 4))
    mr3s = torch.tensor([[0, -1, 0, 0], 
                         [1, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0]], dtype=torch.float, device=rot.device).expand((b, 4, 4))
    mr2s = torch.tensor([[0, 0, 1, 0], 
                         [0, 0, 0, 0], 
                         [-1, 0, 0, 0], 
                         [0, 0, 0, 0]], dtype=torch.float, device=rot.device).expand((b, 4, 4))
    mr1s = torch.tensor([[0, 0, 0, 0], 
                         [0, 0, -1, 0], 
                         [0, 1, 0, 0], 
                         [0, 0, 0, 0]], dtype=torch.float, device=rot.device).expand((b, 4, 4))

    rot_mat_0 = mr1c*torch.cos(rot_0) + mr1s*torch.sin(rot_0) + m111
    rot_mat_1 = mr2c*torch.cos(rot_1) + mr2s*torch.sin(rot_1) + m122
    rot_mat_2 = mr3c*torch.cos(rot_2) + mr3s*torch.sin(rot_2) + m133

    rot_mat = torch.einsum("bij,bjk,bkl->bil", rot_mat_0, rot_mat_1, rot_mat_2)

    return rot_mat

def Rt_from_euler_and_trans(euler, translation, invert=False, angle_normalize_factor = None):
    # type: (torch.Tensor, torch.Tensor, bool, Optional[float]) -> torch.Tensor
    """Convert the network's (euler, translation) output into a 4x4 matrix

    ## Parameters:

    - euler: (b, 3)
    - translation: (b, 3)
    - angle_normalize_factor: If not None, each of euler angle will be calculated tanh and multiplied with angle_normalize_factor. If None, magnitude of vec will be respected as angle.

    ## Return:

    (b, 4, 4)
    """
    R = Rt_from_euler(euler, normalize_factor=angle_normalize_factor)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = Rt_from_trans(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def Rt_to_euler_trans_multi(Rt: torch.Tensor) -> torch.Tensor:
    '''
    ## Parameters

    - Rt: matrix [R | t] with shape (3, 4), (4, 4) or (N, 3, 4), (N, 4, 4). If dim is 2, return shape (6,) Tensor. If dim is 3, convert each of matrix, and return (N, 6)
    '''
    if len(Rt.shape) == 3:
        poses = torch.zeros((Rt.shape[0], 6))
        for i in range(Rt.shape[0]):
            poses[i] = Rt_to_euler_trans(Rt[i])
        return poses
    elif len(Rt.shape) == 2:
        return Rt_to_euler_trans(Rt)
    else:
        raise Exception("shape of argument Rt is unsupported.")

def Rt_to_euler_trans(Rt: torch.Tensor):
# Ground truth pose is present as [R | t] 
# R: Rotation Matrix, t: translation vector
# transform matrix to angles
    # Rt = np.reshape(np.array(Rt), (3,4))
    '''
    ## Parameters

    - Rt: matrix [R | t] with shape (3, 4), (4, 4). return shape (6,) Tensor [t, theta]
    '''
    
    t = Rt[:3,-1]
    R = Rt[:3,:3]

    assert(is_rotation_matrix(R))
    
    theta = R_to_euler(R)

    pose = torch.cat((t, theta)) # type: torch.Tensor
    assert(pose.shape == (6,))
    # pose = torch.cat((theta, t, R.flatten()))
    # assert(pose.shape == (15,))
    return pose
    
def normalize_angle_delta(angle: torch.Tensor):
    PI = torch.tensor(np.pi, device=angle.device)
    if(angle > PI):
        angle = angle - 2 * PI
    elif(angle < -PI):
        angle = 2 * PI + angle
    return angle

def reverse_Rt_matrix(mat: torch.Tensor) -> torch.Tensor:
    '''
    reverse Rt matrix

    ## Parameters

    - mat: Rt matrix. Tensor with shape (..., 3, 4) or (..., 4, 4)

    ## Return
    same shape as input
    '''
    shape = mat.shape
    # shape0 = shape[:-2]
    shape1 = shape[-2:]
    mat = torch.reshape(mat, (-1,)+shape1)
    mat_rev = torch.clone(mat)

    mat_rev[:, :3,:3] = mat[:, :3, :3].permute((0, 2, 1))
    mat_rev[:, :3, 3:] = -torch.matmul(mat_rev[:, :3, :3], mat[:, :3, 3:])

    mat_rev = torch.reshape(mat_rev, shape)

    return mat_rev

# def reverse_Rt_matrix_ori(mat: np.ndarray):
#     '''
#     reverse Rt matrix

#     ## Parameters

#     - mat: Rt matrix. NDArray with shape (..., 3, 4) or (..., 4, 4)
#     '''
#     shape = mat.shape
#     # shape0 = shape[:-2]
#     shape1 = shape[-2:]
#     mat = np.reshape(mat, (-1,)+shape1)
#     mat_rev = np.copy(mat)
#     for i in range(mat_rev.shape[0]):
#         mat_rev[i, :3,:3] = np.transpose(mat[i, :3, :3])
#         mat_rev[i, :3, 3:] = -np.matmul(mat_rev[i, :3, :3], mat[i, :3, 3:])

#     mat_rev = np.reshape(mat_rev, shape)

#     return mat_rev

def pose_abs_to_rel(mat: torch.Tensor) -> torch.Tensor:
    '''
    got (.., N, 3, 4) or (.., N, 4, 4) absolute pose, return relative pose.
    '''
    T_curr_to_last = torch.clone(mat)
    T_curr = T_curr_to_last[...,1:,:,:]
    T_last = T_curr_to_last[...,:-1,:,:]
    T_last_rev = torch.inverse(T_last)
    T_curr_to_last[...,1:,:,:] = torch.matmul(T_last_rev, T_curr)
    T_curr_to_last[...,0,:3,:3] = torch.eye(3)
    return T_curr_to_last

def pose_abs_to_first(mat: torch.Tensor) -> torch.Tensor:
    '''
    got (N, 3, 4) or (N, 4, 4) absolute pose, return relative pose related to the first frame.
    '''
    T_curr_to_first = torch.clone(mat)
    T_first = T_curr_to_first[...,0:1,:,:]
    T_first_rev = torch.inverse(T_first)
    T_curr_to_first = torch.matmul(T_first_rev, T_curr_to_first)
    return T_curr_to_first

def pose_rel_to_abs(mat: torch.Tensor, init: torch.Tensor=None) -> torch.Tensor:
    '''
    got (N, 4, 4) relative pose, return absolute pose.

    ## Parameters:

    - mat: (..., N, 4, 4)
    - init: (..., 1, 4, 4)
    '''
    mat = mat.permute((mat.dim()-3,)+tuple(range(0,mat.dim()-3))+tuple(range(mat.dim()-2, mat.dim())))
    acc = torch.stack(list(accumulate(mat, lambda acc, y: torch.matmul(acc, y))))
    
    if init is not None:
        init = init.permute((init.dim()-3,)+tuple(range(0,init.dim()-3))+tuple(range(init.dim()-2, init.dim())))
        acc = torch.matmul(init, acc)
    
    mat = mat.permute(tuple(range(1,mat.dim()-2))+(0,)+tuple(range(mat.dim()-2, mat.dim())))
    return acc

def seq_adjacent_concat(x):
    # type: (torch.Tensor) -> torch.Tensor
    '''
    ## Parameters:

    - x: shape (batch, seq, *)

    ## Return

    concatenate the former and the latter frames
    '''
    x = torch.cat(( x[:, :-1], x[:, 1:]), dim=2)
    return x

def seq_first_concat(x):
    # type: (torch.Tensor) -> torch.Tensor
    '''
    ## Parameters:

    - x: shape (batch, seq, *)

    ## Return

    concatenate the former and the latter frames
    '''
    x_former = x[:, :1].expand((x.shape[0], x.shape[1]-1)+x.shape[2:])
    x_latter = x[:, 1:]
    x = torch.cat((x_former, x_latter), dim=2)
    return x

CONCAT_TYPE_DICT = (seq_adjacent_concat, seq_first_concat)
    
def get_flownet_update_dict(model_dict, pretrained_state_dict, update_convlstm=True):
    # type: (OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor], bool) -> OrderedDict[str, torch.Tensor]
    '''
    flownet pretrained network is from DeepVO
    '''
    update_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_dict}

    if update_convlstm:
        for k, v in pretrained_state_dict.items():

            if update_convlstm and (k.startswith("conv3_1.0") or k.startswith("conv4_1.0") or k.startswith("conv5_1.0")):
                n_str = k[4]
                k_model_dict = "convlstm{}_1.0.cell_list.0.conv.weight".format(n_str)
                v_model_dict = torch.rand(model_dict[k_model_dict].shape, device=v.device, dtype=v.dtype)
                v_model_dict[:v.shape[0], :v.shape[1],:,:] = v
                update_dict[k_model_dict] = v_model_dict
            
            if update_convlstm and (k.startswith("conv3_1.1") or k.startswith("conv4_1.1") or k.startswith("conv5_1.1")):
                n_str = k[4]
                name_str = k[10:]
                k_model_dict = "convlstm{}_1.2.{}".format(n_str, name_str)
                update_dict[k_model_dict] = v

    return update_dict

## Below is based on Monodepth2

def disp_to_depth(disp, min_depth, max_depth):
    # type: (torch.Tensor, float, float) -> torch.Tensor
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def Rt_from_axisangle_and_trans(axisangle, translation, invert=False, angle_normalize_factor = None):
    # type: (torch.Tensor, torch.Tensor, bool, Optional[float]) -> torch.Tensor
    """Convert the network's (axisangle, translation) output into a 4x4 matrix

    ## Parameters:

    - axisangle: (b, 3)
    - translation: (b, 3)
    - angle_normalize_factor: If not None, each of euler angle will be calculated tanh and multiplied with angle_normalize_factor. If None, magnitude of vec will be respected as angle.

    ## Return:

    (b, 4, 4)
    """
    R = Rt_from_axisangle(axisangle, normalize_factor=angle_normalize_factor)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = Rt_from_trans(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def Rt_from_trans(translation_vector):
    # type: (torch.Tensor) -> torch.Tensor
    """Convert a translation vector into a 4x4 transformation matrix

    ## Parameters:

    - translation_vector: (b, 3) or tensor that can be represent as (b, 3)

    ## Return:

    (b, 4, 4)
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def Rt_from_axisangle(vec, normalize_factor = None):
    # type: (torch.Tensor, Optional[float]) -> torch.Tensor
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3

    ## Parameters:

    - vec: axisangle, (b, 3) or tensor that can be represent as (b, 3)
    - normalize_factor: If not None, magnitude of vec will be calculated tanh and multiplied with normalize_factor. If None, magnitude of vec will be respected as angle.

    ## Return:

    (b, 4, 4)
    """
    vec = vec.reshape((-1, 1, 3))
    if normalize_factor is None:
        angle = torch.norm(vec, 2, 2, True)
    else:
        angle = torch.tanh(torch.norm(vec, 2, 2, True)) * normalize_factor
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


# class BackprojectDepth(nn.Module):
#     """Layer to transform a depth image into a point cloud
#     """
#     def __init__(self, batch_size, height, width):
#         super(BackprojectDepth, self).__init__()

#         self.batch_size = batch_size
#         self.height = height
#         self.width = width

#         meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
#         self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
#         self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
#                                       requires_grad=False)

#         self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
#                                  requires_grad=False)

#         self.pix_coords = torch.unsqueeze(torch.stack(
#             [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
#         self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
#         self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
#                                        requires_grad=False)

#     def forward(self, depth, inv_K):
#         cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
#         cam_points = depth.view(self.batch_size, 1, -1) * cam_points
#         cam_points = torch.cat([cam_points, self.ones], 1)

#         return cam_points


# class Project3D(nn.Module):
#     """Layer which projects 3D points into a camera with intrinsics K and at position T
#     """
#     def __init__(self, batch_size, height, width, eps=1e-7):
#         super(Project3D, self).__init__()

#         self.batch_size = batch_size
#         self.height = height
#         self.width = width
#         self.eps = eps

#     def forward(self, points, K, T):
#         if K.shape[1]==3:
#             K_t = torch.eye(4, device=K.device).unsqueeze(0).expand((K.shape[0], 4, 4))
#             K_t[:, :3, :3] = K
#             K = K_t
#         P = torch.matmul(K, T)[:, :3, :]

#         cam_points = torch.matmul(P, points)

#         pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
#         pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
#         pix_coords = pix_coords.permute(0, 2, 3, 1)
#         pix_coords[..., 0] /= self.width - 1
#         pix_coords[..., 1] /= self.height - 1
#         pix_coords = (pix_coords - 0.5) * 2
#         return pix_coords

def reproject(depth, T, K, K_inv=None, regularize=True, coors=None, return_depth=False):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool, Optional[torch.Tensor], bool) -> torch.Tensor
    '''
    ## Parameters:

    - depth: (b, c, h, w)
    - T: (b, 4, 4). Transformation from depth frame to another.
    - K: (b, 3, 3)
    - K_inv: (b, 3, 3). Computed by torch.inverse if None provided.
    - regularize: convert return coordinates to [-1, 1] if True, original coordinates if False.
    - coors: (1, h, w, 3, 1) or (b, h, w, 3, 1), with vertical vector representing homogeneous coordinate of each pixel. If None, this function will generate automatically.
    '''
    EPS = 1e-7

    assert(depth.shape[1]==1)
    if coors is not None:
        assert(len(coors.shape)==5)
        assert(coors.shape[0]==1 or coors.shape[0]==depth.shape[0])
        assert(coors.shape[1]==depth.shape[2] and coors.shape[2]==depth.shape[3])
        assert(coors.shape[3]==3 and coors.shape[4]==1)

    if K_inv==None:
        K_inv = K.inverse()

    height = depth.shape[2]
    width = depth.shape[3]

    # construct coordinates
    if coors is None:
        grid_i, grid_j = torch.meshgrid(torch.arange(0, height, device=depth.device), torch.arange(0, width, device=depth.device))
        ones = torch.ones_like(grid_i)
        coors = torch.stack((grid_j, grid_i, ones), dim=-1).unsqueeze(-1).unsqueeze(0).to(dtype=torch.float32) # (1, h, w, 3, 1) # use grid_j, grid_i for that x axis is horizontal, y axis is vertical, and z axis go forward

    # back project
    K = K.unsqueeze(1).unsqueeze(1) # (b, 1, 1, 3, 3)
    K_inv = K_inv.unsqueeze(1).unsqueeze(1) # (b, 1, 1, 3, 3)
    T = T.unsqueeze(1).unsqueeze(1) # (b, 1, 1, 4, 4)

    depth = depth.squeeze(1).unsqueeze(-1).unsqueeze(-1) # (b, h, w, 1, 1)
    points = torch.matmul(K_inv, coors) * depth # (b, h, w, 3, 1)

    # transform and project
    ones_4 = torch.ones_like(depth)
    points_2 = torch.cat((points, ones_4), dim=-2) # (b, h, w, 4, 1)
    
    points_2 = torch.matmul(T, points_2)
    points_2 = torch.matmul(K, points_2[:,:,:,:3,:]) # (b, h, w, 3, 1)

    coors_2 = torch.zeros_like(points_2) # (b, h, w, 3, 1)
    coors_2[:,:,:,:2,:] = points_2[:,:,:,:2,:] / (points_2[:,:,:,2:3,:] + EPS)
    coors_2[:,:,:,2:,:] = points_2[:,:,:,2:,:]
    coors_2 = coors_2.squeeze(-1) # (b, h, w, 3)

    if regularize:
        coors_2[..., 0] /= width - 1
        coors_2[..., 1] /= height - 1
        coors_2[..., :2] = (coors_2[..., :2] - 0.5) * 2

    if return_depth:
        return coors_2 # (b, h, w, 3)
    else:
        return coors_2[..., :2] # (b, h, w, 2)

class Reproject:
    def __init__(self, height, width, device=None):
        self.height = height
        self.width = width
        grid_i, grid_j = torch.meshgrid(torch.arange(0, height, device=device), torch.arange(0, width, device=device))
        ones = torch.ones_like(grid_i)
        self.coors = torch.stack((grid_j, grid_i, ones), dim=-1).unsqueeze(-1).unsqueeze(0).to(dtype=torch.float32) # (1, h, w, 3, 1) # use grid_j, grid_i for that x axis is horizontal, y axis is vertical, and z axis go forward

    def __call__(self, depth, T, K, K_inv=None, regularize=True):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool) -> torch.Tensor
        '''
        ## Parameters:

        - depth: (b, c, h, w)
        - T: (b, 4, 4). Transformation from depth frame to another.
        - K: (b, 3, 3)
        - K_inv: (b, 3, 3). Computed by torch.inverse if None provided.
        - regularize: convert return coordinates to [-1, 1] if True, original coordinates if False.
        '''
        return reproject(depth, T, K, K_inv, regularize, self.coors)

    def to(self, dtype_or_device):
        self.coors = self.coors.to(dtype_or_device)

    def cuda(self, device = None, non_blocking = False):
        self.coors = self.coors.cuda(device, non_blocking)

    def cpu(self):
        self.coors = self.coors.cpu()

def sample(img, coors, mode="bilinear", padding_mode="border", align_corners=None):
    # type: (torch.Tensor, torch.Tensor, str, str, Optional[bool]) -> torch.Tensor
    '''
    ## Parameters:

    - coors: (b, h, w, 2), value from [-1, 1]
    - img: (b, c, h, w)
    - mode (str): interpolation mode to calculate output values
        ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``
        Note: ``mode='bicubic'`` supports only 4-D input.
        When ``mode='bilinear'`` and the input is 5-D, the interpolation mode
        used internally will actually be trilinear. However, when the input is 4-D,
        the interpolation mode will legitimately be bilinear.
    - padding_mode (str): padding mode for outside grid values
        ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
    - align_corners (bool, optional): Geometrically, we consider the pixels of the
        input  as squares rather than points.
        If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
        to the center points of the input's corner pixels. If set to ``False``, they
        are instead considered as referring to the corner points of the input's corner
        pixels, making the sampling more resolution agnostic.
        This option parallels the ``align_corners`` option in
        :func:`interpolate`, and so whichever option is used here
        should also be used there to resize the input image before grid sampling.
        Default: ``False``
    '''
    return F.grid_sample(img, coors, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def sample_from_depth(depth, img, T, K, K_inv=None):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]) -> torch.Tensor
    '''
    ## Parameters:

    - depth: (b, c, h, w)
    - img: (b, c, h, w)
    - T: (b, 4, 4). Transformation from depth frame to img frame.
    - K: (b, 3, 3)
    - K_inv: (b, 3, 3). Computed by torch.inverse if None provided.
    '''
    coors = reproject(depth, T, K, K_inv)
    return sample(img, coors)

def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")

def pil_to_tensor(pil_img):
    transform = transforms.ToTensor()
    return transform(pil_img).to(dtype=torch.float32)

def tensor_to_pil(tensor_img):
    transform = transforms.ToPILImage()
    return transform(tensor_img)

# Below is based on article Deep Visual Odometry with Adaptive Memory

def cosine_similarity_per_channel(x, y):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    '''
    compute cosine similarity per channel

    ## Parameters:
    
    -x, y: (b, c, h, w)

    ## Return

    (b, c)
    '''
    assert(x.shape == y.shape)
    shapes = x.shape
    sim = F.cosine_similarity(x.reshape(shapes[:2]+(-1, )), y.reshape(shapes[:2]+(-1, )))
    return sim

def spatial_attention(m, o):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    '''
    calculate cosine similarity per channel between m and o, and weight m (element-wise product)

    ## Parameters:

    - m, o: (b, c, h, w)

    ## Return:

    (b, c, h, w)
    '''
    beta = cosine_similarity_per_channel(m, o)
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    m_p = beta * m
    return m_p

def spatio_temporal_attention(m, o):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    '''
    ## Parameters:

    - m: (b, c, h, w)
    - o: (c, h, w)

    ## Return

    (c, h, w)
    '''
    assert(m.shape[1:]==o.shape)
    shapes = m.shape

    m_p = spatial_attention(m, o.unsqueeze(0).expand_as(m))
    w = F.cosine_similarity(m_p.reshape((shapes[0], -1)), o.reshape((1, -1)).expand((shapes[0], -1)))
    
    def normalize(w):
        # type: (torch.Tensor) -> torch.Tensor
        w_exp = w.exp()
        alpha = w_exp/w_exp.sum()
        return alpha

    alpha = normalize(w)

    m_p = torch.sum(alpha.reshape((-1, 1, 1, 1)) * m_p, dim=0)

    return m_p

def inner_channel_corelation2d(input, weight, dim_is_padding=None):
    if dim_is_padding is None:
        dim_is_padding = (True, True)
    else:
        assert(isinstance(dim_is_padding, Iterable) and len(dim_is_padding) == 2)
    pad = []
    for i in range(input.dim()-2):
        pad.extend((input.shape[-1-i]//2,)*2 if i < len(dim_is_padding) and dim_is_padding[i] else (0,0))
    input = F.pad(input, pad, "circular")
    heading_shapes = input.shape[:-2]
    input = input.flatten(0,-3)
    weight = weight.flatten(0,-3)
    ret = []
    for i in range(input.shape[0]):
        ret.append(F.conv2d(input[i:i+1].unsqueeze(0), weight[i:i+1].unsqueeze(0)))
    ret = torch.cat(ret, 0)
    if dim_is_padding[0]:
        ret = ret[:,:,:,:-1]
    if dim_is_padding[1]:
        ret = ret[:,:,:-1,:]
    ret = ret.reshape(heading_shapes + ret.shape[-2:])
    return ret
