import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import numpy as np
import pybullet as p
import pybullet_data

def convert_opencv_to_pybullet(opencv_transform):
    """
    Convert a transformation matrix from OpenCV coordinates to PyBullet coordinates.
    
    Args:
    opencv_transform (np.ndarray): A 4x4 transformation matrix in OpenCV coordinates.
    
    Returns:
    np.ndarray: A 4x4 transformation matrix in PyBullet coordinates.
    """
    # Create a matrix to convert from OpenCV to PyBullet coordinate system
    opencv_to_pybullet = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # Convert the transformation matrix
    pybullet_transform = opencv_to_pybullet @ opencv_transform @ np.linalg.inv(opencv_to_pybullet)
    
    return pybullet_transform

def convert_opengl_to_pybullet(opengl_transform):
    """
    Convert a transformation matrix from OpenGL coordinates to PyBullet coordinates.
    
    Args:
    opencv_transform (np.ndarray): A 4x4 transformation matrix in OpenGL coordinates.
    
    Returns:
    np.ndarray: A 4x4 transformation matrix in PyBullet coordinates.
    """
    # Create a matrix to convert from OpenCV to PyBullet coordinate system
    opengl_to_pybullet = np.array([
        [0, 0, -1, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    
    # Convert the transformation matrix
    pybullet_transform = opengl_to_pybullet @ opengl_transform @ np.linalg.inv(opengl_to_pybullet)
    
    return pybullet_transform
