import os, inspect
import pybullet as p
import sys
import numpy as np
import json
sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# robot
sys.path.append("/home/exx/Yubin/deep_mimic")
sys.path.append("/home/exx/Yubin/deep_mimic/mocap")
import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85, Panda
from pybullet_utils.bullet_client import BulletClient
from utils.transform_utils import *
from utils.grasp_utils import *
# from utils.debug_utils import *

from scipy.spatial.transform import Rotation as R

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

def draw_sphere_marker(bc, position, radius=0.02, color=[1,0,0,1]):
  vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
  marker_id = bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

def draw_frame(bc, position, quaternion=[0, 0, 0, 1]):
    m = R.from_quat(quaternion).as_matrix()
    x_vec = m[:, 0]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for color, column in zip(colors, range(3)):
        vec = m[:, column]
        from_p = position
        to_p = position + (vec * 0.1)
        bc.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)

def reset_robot(bc, robot, q_robot):
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        bc.resetJointState(robot.id, joint_id, q_robot[i])
    bc.stepSimulation()

def move_robot(bc, robot, q_robot):
  for _ in range(500):
    for i, joint_id in enumerate(robot.arm_controllable_joints):
      bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i],
                                force=robot.joints[joint_id].maxForce, maxVelocity=robot.joints[joint_id].maxVelocity)
    bc.stepSimulation()

def create_transformation_matrix(control_points):
  matrices = []
  for point in control_points:
    T = compute_matrix(translation=point, rotation=[0,0,0,1])
    matrices.append(T)
  return matrices

# load environment
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.setGravity(0, 0, -9.8) 
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0))

# load UR5 robot
ur5 = UR5Robotiq85(bc, (0, 0, 0), (0, 0, 0))
ur5.load()
ur5.reset()
for i in range(bc.getNumJoints(ur5.id)):
    print(bc.getJointInfo(ur5.id, i))

# load Panda robot
panda = Panda(bc, (0.5, 0, 0), (0, 0, 0))
panda.load()
panda.reset()
for i in range(bc.getNumJoints(panda.id)):
    print(bc.getJointInfo(panda.id, i))

# get UR5 eef pose
ur5_ee_link = bc.getLinkState(ur5.id, ur5.eef_id)[:2]
draw_frame(bc, position=ur5_ee_link[0], quaternion=ur5_ee_link[1])
ur5_ee_position = [ur5_ee_link[0][0], ur5_ee_link[0][1], ur5_ee_link[0][2]-0.13]
ur5_ee_orientation = ur5_ee_link[1]
draw_frame(bc, position=ur5_ee_position, quaternion=ur5_ee_orientation)

# get Panda eef pose
panda_ee_link = bc.getLinkState(panda.id, panda.eef_id)[:2]
panda_orientation = rotate_quaternion_by_axis(panda_ee_link[1], axis='y', degrees=-90)
panda_orientation = rotate_quaternion_by_axis(panda_orientation, axis='x', degrees=-90)
draw_frame(bc, position=panda_ee_link[0], quaternion=panda_orientation)


print('done')