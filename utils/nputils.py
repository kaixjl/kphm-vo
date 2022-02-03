# Based on DeepVO-pytorch
import numpy as np
import math
from . import nputils2

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
    
def Rt_to_euler_trans_multi(Rt: np.ndarray) -> np.ndarray:
    '''
    ## Parameters

    - Rt: matrix [R | t] with shape (3, 4), (4, 4) or (N, 3, 4), (N, 4, 4). If dim is 2, return shape (6,) ndarray. If dim is 3, convert each of matrix, and return (N, 6)
    '''
    if len(Rt.shape) == 3:
        poses = np.zeros((Rt.shape[0], 6))
        for i in range(Rt.shape[0]):
            poses[i] = Rt_to_euler_trans(Rt[i])
        return poses
    elif len(Rt.shape) == 2:
        return Rt_to_euler_trans(Rt)
    else:
        raise Exception("shape of argument Rt is unsupported.")

def Rt_to_euler_trans(Rt: np.ndarray) -> np.ndarray:
# Ground truth pose is present as [R | t] 
# R: Rotation Matrix, t: translation vector
# transform matrix to angles
    # Rt = np.reshape(np.array(Rt), (3,4))
    '''
    ## Parameters

    - Rt: matrix [R | t] with shape (3, 4), (4, 4). return shape (6,) ndarray [t, theta]
    '''
    
    t = Rt[:3,-1]
    R = Rt[:3,:3]

    assert(isRotationMatrix(R))
    
    theta = R_to_euler(R)
    
    pose = np.concatenate((t, theta))
    assert(pose.shape == (6,))
    # pose = np.concatenate((theta, t, R.flatten()))
    # assert(pose_15.shape == (15,))
    return pose

def R_from_euler(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_x, np.dot( R_y, R_z ))
    return R
    
# def R_to_euler(matrix):
#       默认matrix是X->Y->Z，即Mp=ZYXp    
#     # y-x-z Tait–Bryan angles intrincic
#     # the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py
    
#     i = 2
#     j = 0
#     k = 1
#     repetition = 0
#     frame = 1
#     parity = 0
    

#     M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
#     if repetition:
#         sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
#         if sy > _EPS:
#             ax = math.atan2( M[i, j],  M[i, k])
#             ay = math.atan2( sy,       M[i, i])
#             az = math.atan2( M[j, i], -M[k, i])
#         else:
#             ax = math.atan2(-M[j, k],  M[j, j])
#             ay = math.atan2( sy,       M[i, i])
#             az = 0.0
#     else:
#         cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
#         if cy > _EPS:
#             ax = math.atan2( M[k, j],  M[k, k])
#             ay = math.atan2(-M[k, i],  cy)
#             az = math.atan2( M[j, i],  M[i, i])
#         else:
#             ax = math.atan2(-M[j, k],  M[j, j])
#             ay = math.atan2(-M[k, i],  cy)
#             az = 0.0

#     if parity:
#         ax, ay, az = -ax, -ay, -az
#     if frame:
#         ax, az = az, ax
#     return np.array((ax, ay, az), dtype=matrix.dtype)

def R_to_euler(matrix):
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
    M = np.asarray(matrix)
    try:
        cy_thresh = np.finfo(M.dtype).eps * 4
    except ValueError:
        cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return np.array([x, y, z])
    pass
    
def normalize_angle_delta(angle):
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle

def reverse_Rt_matrix(mat: np.ndarray) -> np.ndarray:
    '''
    reverse Rt matrix

    ## Parameters

    - mat: Rt matrix. NDArray with shape (..., 3, 4) or (..., 4, 4)

    ## Return
    same shape as input
    '''
    shape = mat.shape
    # shape0 = shape[:-2]
    shape1 = shape[-2:]
    mat = np.reshape(mat, (-1,)+shape1)
    mat_rev = np.copy(mat)

    mat_rev[:, :3,:3] = np.transpose(mat[:, :3, :3], axes=(0, 2, 1))
    mat_rev[:, :3, 3:] = -np.matmul(mat_rev[:, :3, :3], mat[:, :3, 3:])

    mat_rev = np.reshape(mat_rev, shape)

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

def pose_abs_to_rel(mat: np.ndarray) -> np.ndarray:
    '''
    got (N, 3, 4) or (N, 4, 4) absolute pose, return relative pose.
    '''
    T_curr_to_last = np.copy(mat)
    T_curr = T_curr_to_last[1:,:,:]
    T_last = T_curr_to_last[:-1,:,:]
    T_last_rev = reverse_Rt_matrix(T_last)
    T_curr_to_last[1:,:,:] = np.matmul(T_last_rev, T_curr)
    T_curr_to_last[0,:3,:3] = np.eye(3)
    return T_curr_to_last

def pose_abs_to_first(mat: np.ndarray) -> np.ndarray:
    '''
    got (N, 3, 4) or (N, 4, 4) absolute pose, return relative pose.
    '''
    T_curr_to_first = np.copy(mat)
    T_first = T_curr_to_first[0:1,:,:]
    T_first_rev = reverse_Rt_matrix(T_first)
    T_curr_to_first = np.matmul(T_first_rev, T_curr_to_first)
    return T_curr_to_first

def adjacent_concat(x):
    '''
    ## Parameters:

    - x: shape (batch, seq, *)
    '''
    x = np.concatenate(( x[:, :-1], x[:, 1:]), dim=2)
    return x