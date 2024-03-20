# ur5, pybullet
import os, inspect
import os.path as osp
import pybullet as p
import math
import sys
sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
print('parentdir: ', parentdir)
os.sys.path.insert(0, parentdir)

import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85
from pybullet_utils.bullet_client import BulletClient
import time
import numpy as np

# humanoid
# from deep_mimic.env.motion_capture_data import MotionCaptureData
from env.motion_capture_data import MotionCaptureData
from humanoid import Humanoid
from humanoid import HumanoidPose

# ramp
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower


class MPPI_H_Clamp():
    def __init__(self, eef_to_cp, human_arm_lower_limits, human_arm_upper_limits, robot_current_joint_angles):
        # 2nd BC server
        self.bc_second = BulletClient(connection_mode=p.DIRECT)
        self.bc_second.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc_second.configureDebugVisualizer(self.bc_second.COV_ENABLE_Y_AXIS_UP, 1)

        # load humanoid in the 2nd server
        motionPath = 'data/Greeting.json'
        motion = MotionCaptureData()
        motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc_second, motion, [0, 0.3, 0])

        # load robot in the 2nd server
        y2zOrn = self.bc_second.getQuaternionFromEuler((-1.57, 0, 0))
        self.robotID2 = self.bc_second.loadURDF('resources/panda/panda.urdf', (-1.0, -0.04, 0), y2zOrn, useFixedBase=True)
        self.robot_eef = 11
        self.numJoints = 7
        for i, joint_angle in enumerate(robot_current_joint_angles):
            self.bc_second.setJointMotorControl2(self.robotID2, i, p.POSITION_CONTROL, joint_angle)

        # initialize T_eef_cp (constant)
        self.eef_to_cp = eef_to_cp

        # human arm joint limits
        self.human_arm_lower_limits = human_arm_lower_limits
        self.human_arm_upper_limits = human_arm_upper_limits
    
    def clamp_human_joints(self, q_R_list):
        # print('q_R_list before clamp: ', q_R_list)
        for q_R in q_R_list:
            # get eef pose from q_R
            for i in range(self.numJoints):
                self.bc_second.setJointMotorControl2(self.robotID2, i, p.POSITION_CONTROL, q_R[i])
            eef_to_world = self.bc_second.getLinkState(self.robotID2, self.robot_eef)

            # get cp pose
            cp_to_eef = self.bc_second.invertTransform(self.eef_to_cp[0], self.eef_to_cp[1])
            cp_to_world = self.bc_second.multiplyTransforms(cp_to_eef[0], cp_to_eef[1],
                                                            eef_to_world[0], eef_to_world[1])

            # IK -> get human joint angles
            rightShoulder = 3
            rightElbow = 4
            q_H = self.bc_second.calculateInverseKinematics(self.humanoid._humanoid, rightElbow, targetPosition=cp_to_world[0], targetOrientation=cp_to_world[1])
            
            # TODO check q_H with joint limits, clamp
            clamped_q_H = []
            for i in range(len(q_H)):
                if q_H[i] < self.human_arm_lower_limits[i]:
                    clamped_q_H.append(self.human_arm_lower_limits[i])
                elif q_H[i] > self.human_arm_upper_limits[i]:
                    clamped_q_H.append(self.human_arm_upper_limits[i])
                else:
                    clamped_q_H.append(q_H[i])

            # # move humanoid in the 2nd server, get new cp pose
            self.bc_second.setJointMotorControlMultiDof(self.humanoid._humanoid, rightShoulder, controlMode=p.POSITION_CONTROL, targetPosition=q_H[:3])
            self.bc_second.setJointMotorControl2(self.humanoid._humanoid, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=q_H[3])
            cp_to_world = self.bc_second.getLinkState(self.humanoid._humanoid, rightElbow)

            # get new eef pose
            eef_to_world = self.bc_second.multiplyTransforms(self.eef_to_cp[0], self.eef_to_cp[1],
                                                            cp_to_world[0], cp_to_world[1])

            # IK -> get new robot joint angles
            q_R = self.bc_second.calculateInverseKinematics(self.robotID2, self.robot_eef, eef_to_world[0], eef_to_world[1])

        # print('q_R_list after clamp: ', q_R_list)
        return q_R_list
