# ur5, pybullet
import os, inspect
import os.path as osp
import pybullet as p
import math
import sys
sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85
from pybullet_utils.bullet_client import BulletClient
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from utils.debug_utils import *
from utils.transform_utils import *

# point cloud
import open3d as o3d
from utils.point_cloud_utils import *

# UR5 parameters
LINK_SKELETON_NAME = [
    'shoulder_link',
    'upper_arm_link',
    'forearm_link',
    'wrist_1_link',
    'wrist_2_link',
    'wrist_3_link',
    'ee_link',
]
LINK_SKELETON_JOINT_NUM = [1, 2, 3, 4, 5, 6, 7]

# debug helper functions
def draw_sphere_marker(bc, position, radius, color=[1,0,0,1]):
  vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
  marker_id = bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

def move_robot(bc, robot, q_robot):
  for _ in range(500):
    for i, joint_id in enumerate(robot.arm_controllable_joints):
      bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i],
                                force=robot.joints[joint_id].maxForce, maxVelocity=robot.joints[joint_id].maxVelocity)
    bc.stepSimulation()

def visualize_point_cloud(pcd):
    pc_ply = o3d.geometry.PointCloud()
    pc_ply.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pc_ply])

# function to get point cloud from visual shape vertices
def get_point_cloud_from_visual_shapes(body_id, link_ids):
    point_cloud_dict = {
        'shoulder_link': [],
        'upper_arm_link': [],
        'forearm_link': [],
        'wrist_1_link': [],
        'wrist_2_link': [],
        'wrist_3_link': [],
        'ee_link': [],
    }
    point_cloud = []
    for link_id in link_ids:
        visual_data = p.getCollisionShapeData(body_id, link_id)
        for data in visual_data:
            if data[2] in [p.GEOM_MESH, p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
                position = data[5]
                orientation = data[6]

                if data[2] == p.GEOM_MESH:
                    mesh_file = data[4].decode("utf-8")
                    mesh_scale = data[3]
                    vertices = p.getMeshData(body_id, link_id)[1]
                elif data[2] == p.GEOM_BOX:
                    half_extents = np.array(data[3])
                    vertices = generate_box_vertices(half_extents, position, orientation, resolution=1)
                elif data[2] == p.GEOM_SPHERE:
                    radius = data[3][0]
                    vertices = generate_sphere_vertices(radius, position, orientation, resolution=1)
                elif data[2] == p.GEOM_CYLINDER:
                    height = data[3][0]
                    radius = data[3][1]
                    vertices = generate_cylinder_vertices(radius, height, position, orientation, resolution=1)
                elif data[2] == p.GEOM_CAPSULE:
                    height = data[3][0]
                    radius = data[3][1]
                    vertices = generate_capsule_vertices(radius, height, position, orientation, resolution=1)

                if link_id == -1: 
                    link_state = p.getBasePositionAndOrientation(body_id)
                else:
                    link_state = p.getLinkState(body_id, link_id)

                link_world_pos = link_state[0]
                link_world_ori = link_state[1]
                link_world_transform = p.getMatrixFromQuaternion(link_world_ori)
                link_world_transform = np.array(link_world_transform).reshape(3, 3)
                
                for vertex in vertices:
                    vertex_world = np.dot(link_world_transform, vertex) + np.array(link_world_pos)
                    point_cloud_dict[LINK_SKELETON_NAME[link_id-1]].append(vertex_world)
                    point_cloud.append(vertex_world)
                visualize_point_cloud(point_cloud)

            else:
                print('???wha')
                print(data[2])
    return point_cloud_dict, point_cloud

# load environment
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.setGravity(0, 0, -9.8) 
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0))

# load robot
robot = UR5Robotiq85(bc, (0, 0, 0), (0, 0, 0))
robot.load()
robot.reset()

for i in range(bc.getNumJoints(robot.id)):
  print(bc.getJointInfo(robot.id, i))

point_cloud_dict, point_cloud = get_point_cloud_from_visual_shapes(robot.id, LINK_SKELETON_JOINT_NUM)
for key, value in point_cloud_dict.items():
   print(key, len(point_cloud_dict[key]))
#    draw_sphere_marker(bc, value, radius=0.02)

print('done')

visualize_point_cloud(point_cloud)