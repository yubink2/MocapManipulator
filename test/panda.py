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
from pybullet_ur5.robot import Panda
from pybullet_utils.bullet_client import BulletClient
from utils.transform_utils import *
# from utils.debug_utils import *

from scipy.spatial.transform import Rotation as R


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

# load robot
robot = Panda(bc, (0, 0, 0), (0, 0, 0))
robot.load()
robot.reset()
robot.open_gripper()
for i in range(bc.getNumJoints(robot.id)):
    print(bc.getJointInfo(robot.id, i))

# get link pose
ee_link = bc.getLinkState(robot.id, robot.eef_id)[:2]
robotiq_85_base = bc.getLinkState(robot.id, 8)[:2]
world_to_eef_grasp = [[robotiq_85_base[0][0], robotiq_85_base[0][1], robotiq_85_base[0][2]-0.085], robotiq_85_base[1]]

draw_frame(bc, position=ee_link[0], quaternion=ee_link[1])
# draw_frame(bc, position=robotiq_85_base[0], quaternion=robotiq_85_base[1])
# draw_frame(bc, position=world_to_eef_grasp[0], quaternion=world_to_eef_grasp[1])



print('done')