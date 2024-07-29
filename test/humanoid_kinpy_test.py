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
import torch
import kinpy as kp

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

# # test pybullet IK on humanoid
# q_H = bc.calculateInverseKinematics(humanoid, right_wrist, targetPosition=target_pos)
# print('q_H ', q_H)

# test pytorch IK on humanoid
chain = kp.build_serial_chain_from_urdf(open("./urdf/humanoid_with_rev_scaled.urdf").read(), 
                                        end_link_name="right_wrist",
                                        # root_link_name="right_shoulder_yaw"
                                        )
print(chain)
print(chain.get_joint_parameter_names())

###### pytorch FK
humanoid_base = bc.getBasePositionAndOrientation(humanoid)[:2]
T_world_to_base = compute_matrix(translation=humanoid_base[0], rotation=humanoid_base[1])
# q_H = [0.3, 0.7, -1.0, 0.2]
q_H = [0, 0, 0, 0]
reset_human_arm(q_H)
bc.stepSimulation()

ret = chain.forward_kinematics(q_H, end_only=False)
right_shoulder_yaw_m = T_world_to_base @ ret['right_shoulder_yaw'].get_matrix()
right_elbow_m = T_world_to_base @ ret['right_elbow'].get_matrix()
right_wrist_m = T_world_to_base @ ret['right_wrist'].get_matrix()
draw_frame(right_shoulder_yaw_m[:, :3, 3][0], kp.matrix_to_quaternion(right_shoulder_yaw_m[:, :3, :3][0]))
draw_frame(right_elbow_m[:, :3, 3][0], kp.matrix_to_quaternion(right_elbow_m[:, :3, :3][0]))
draw_frame(right_wrist_m[:, :3, 3][0], kp.matrix_to_quaternion(right_wrist_m[:, :3, :3][0]))
print('done')

