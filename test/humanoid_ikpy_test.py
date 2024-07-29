import os, inspect
import os.path as osp
import pybullet as p
import math
import time
import sys
from scipy.spatial.transform import Rotation as R
sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# robot
sys.path.append("/home/exx/Yubin/deep_mimic")
sys.path.append("/home/exx/Yubin/deep_mimic/mocap")
import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85
from pybullet_utils.bullet_client import BulletClient

# humanoid
from env.motion_capture_data import MotionCaptureData
from humanoid_with_rev import Humanoid
from humanoid_with_rev import HumanoidPose

# kinematics tools
import pytorch_kinematics as pk
import torch
import ikpy.chain

# utils
from utils.transform_utils import *


def draw_frame(position, quaternion=[0, 0, 0, 1]):
    m = R.from_quat(quaternion).as_matrix()
    x_vec = m[:, 0]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for color, column in zip(colors, range(3)):
        vec = m[:, column]
        from_p = position
        to_p = position + (vec * 0.1)
        bc.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)

def move_human_arm(q_human):
    bc.setJointMotorControl2(humanoid, right_shoulder_y, p.POSITION_CONTROL, q_human[0])
    bc.setJointMotorControl2(humanoid, right_shoulder_p, p.POSITION_CONTROL, q_human[1])
    bc.setJointMotorControl2(humanoid, right_shoulder_r, p.POSITION_CONTROL, q_human[2])
    bc.setJointMotorControl2(humanoid, right_elbow, p.POSITION_CONTROL, q_human[3])

def reset_human_arm(q_human):
    bc.resetJointState(humanoid, right_shoulder_y, q_human[0])
    bc.resetJointState(humanoid, right_shoulder_p, q_human[1])
    bc.resetJointState(humanoid, right_shoulder_r, q_human[2])
    bc.resetJointState(humanoid, right_elbow, q_human[3])

def quaternion_xyzw_to_wxyz(quaternion):
    return [quaternion[3], quaternion[0], quaternion[1], quaternion[2]] 

# load environment
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.setGravity(0, 0, -9.8)
humanoid = bc.loadURDF("./urdf/humanoid_with_rev_scaled.urdf",
                        useFixedBase=True)

for j in range(bc.getNumJoints(humanoid)):  
    print(bc.getJointInfo(humanoid, j))

right_shoulder_y = 3
right_shoulder_p = 4
right_shoulder_r = 5
right_shoulder = 6
right_elbow = 7
right_wrist = 8

right_wrist_pose = bc.getLinkState(humanoid, right_wrist)[4:6]
print('right_wrist: ', right_wrist_pose)
target_pos = [right_wrist_pose[0][0], right_wrist_pose[0][1], right_wrist_pose[0][2]]
target_orn = quaternion_xyzw_to_wxyz([0, 0, 0, 1])
# draw_frame(target_pos, target_orn)

# test ikpy FK on humanoid
q_H = [0, 0, 0, 0]
reset_human_arm(q_H)
bc.stepSimulation()

human_chain = ikpy.chain.Chain.from_urdf_file("./urdf/humanoid_with_rev_scaled.urdf")
result = human_chain.forward_kinematics(q_H, full_kinematics=True)
print('done')

