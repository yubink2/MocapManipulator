from scipy.spatial.transform import Rotation as R
import numpy as np
import pybullet as p

def compute_matrix(translation, rotation, rotation_type='quaternion'):
    M = np.zeros((4, 4))
    M[3, 3] = 1.0
    if rotation_type == 'quaternion':
        M[:3, :3] = quaternion_to_matrix(rotation)
    elif rotation_type == 'euler':
        M[:3, :3] = euler_to_matrix(rotation)
    else:
        raise ValueError("rotation_type must be 'quaternion' or 'euler'")
    M[:3, 3] = translation
    return M

def inverse_matrix(matrix):
    R_inv = matrix[:3, :3].T
    t_inv = -R_inv @ matrix[:3, 3]
    M_inv = np.eye(4)
    M_inv[:3, :3] = R_inv
    M_inv[:3, 3] = t_inv
    return M_inv

def quaternion_to_matrix(quaternion):
    return R.from_quat(quaternion).as_matrix()

def euler_to_matrix(euler_angles, sequence='xyz'):
    return R.from_euler(sequence, euler_angles).as_matrix()

def translation_from_matrix(matrix):
    return matrix[:3, 3]

def quaternion_from_matrix(matrix):
    return R.from_matrix(matrix[:3, :3]).as_quat()

def euler_from_matrix(matrix, sequence='xyz'):
    return R.from_matrix(matrix[:3, :3]).as_euler(sequence)