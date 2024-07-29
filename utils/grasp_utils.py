import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

from utils.point_cloud_utils import get_point_cloud_from_collision_shapes_specific_link


def rotate_quaternion_by_axis(quaternion, axis='y', degrees=-90):
    rotation = R.from_euler(axis, degrees, degrees=True).as_quat()
    new_quaternion = R.from_quat(quaternion) * R.from_quat(rotation)
    
    return new_quaternion.as_quat()

def filter_transform_matrices_by_position(matrices, x_range, y_range, z_range):
    """
    Filters transformation matrices based on the user-defined range of positions and returns the indices.
    
    Parameters:
    matrices (numpy array): The array of transformation matrices.
    x_range (tuple): The range (min, max) for the x coordinate.
    y_range (tuple): The range (min, max) for the y coordinate.
    z_range (tuple): The range (min, max) for the z coordinate.
    
    Returns:
    tuple: (filtered_matrices, indices)
           filtered_matrices (numpy array): The filtered transformation matrices.
           indices (list): The indices of the filtered transformation matrices.
    """
    filtered_matrices = []
    indices = []
    for idx, matrix in enumerate(matrices):
        position = matrix[:3, 3]  # Extract the position (x, y, z)
        if (x_range[0] <= position[0] <= x_range[1] and
            y_range[0] <= position[1] <= y_range[1] and
            z_range[0] <= position[2] <= z_range[1]):
            filtered_matrices.append(matrix)
            indices.append(idx)
    return np.array(filtered_matrices), indices





# def convert_opencv_to_pybullet(opencv_transform):
#     """
#     Convert a transformation matrix from OpenCV coordinates to PyBullet coordinates.
    
#     Args:
#     opencv_transform (np.ndarray): A 4x4 transformation matrix in OpenCV coordinates.
    
#     Returns:
#     np.ndarray: A 4x4 transformation matrix in PyBullet coordinates.
#     """
#     # Create a matrix to convert from OpenCV to PyBullet coordinate system
#     opencv_to_pybullet = np.array([
#         [0, 0, 1, 0],
#         [-1, 0, 0, 0],
#         [0, -1, 0, 0],
#         [0, 0, 0, 1]
#     ])
    
#     # Convert the transformation matrix
#     pybullet_transform = opencv_to_pybullet @ opencv_transform @ np.linalg.inv(opencv_to_pybullet)
    
#     return pybullet_transform

# def convert_opengl_to_pybullet(opengl_transform):
#     """
#     Convert a transformation matrix from OpenGL coordinates to PyBullet coordinates.
    
#     Args:
#     opencv_transform (np.ndarray): A 4x4 transformation matrix in OpenGL coordinates.
    
#     Returns:
#     np.ndarray: A 4x4 transformation matrix in PyBullet coordinates.
#     """
#     # Create a matrix to convert from OpenCV to PyBullet coordinate system
#     opengl_to_pybullet = np.array([
#         [0, 0, -1, 0],
#         [-1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 0, 1]
#     ])
    
#     # Convert the transformation matrix
#     pybullet_transform = opengl_to_pybullet @ opengl_transform @ np.linalg.inv(opengl_to_pybullet)
    
#     return pybullet_transform
