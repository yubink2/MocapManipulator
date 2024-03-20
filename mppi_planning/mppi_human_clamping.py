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

# Torch imports
import torch

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
        self.robot = UR5Robotiq85(self.bc_second, (-1.0, 0.35, 0), (-1.57, 0, 0))
        self.robot.load()
        self.robot.reset()
        # for joint_idx, joint_angle in zip(self.robot.arm_controllable_joints, robot_current_joint_angles):
        #     self.bc_second.setJointMotorControl2(self.robot.id, joint_idx, p.POSITION_CONTROL, joint_angle)
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            self.bc_second.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, robot_current_joint_angles[i])

        # initialize T_eef_cp (constant)
        self.eef_to_cp = eef_to_cp

        # human arm joint limits
        self.human_arm_lower_limits = human_arm_lower_limits
        self.human_arm_upper_limits = human_arm_upper_limits
    
    def clamp_human_joints(self, q_R_list, device):
        # print('q_R_list before clamp: ', q_R_list)
        for idx, q_R in enumerate(q_R_list):
            print('q_R before clamping: ', q_R)

            # get eef pose from q_R
            for i, joint in enumerate(self.robot.arm_controllable_joints):
                self.bc_second.setJointMotorControl2(self.robot.id, joint, p.POSITION_CONTROL, q_R[i])
                self.bc_second.stepSimulation()
            eef_to_world = self.bc_second.getLinkState(self.robot.id, self.robot.eef_id)[:2]

            # get cp pose
            cp_to_eef = self.bc_second.invertTransform(self.eef_to_cp[0], self.eef_to_cp[1])
            cp_to_world = self.bc_second.multiplyTransforms(cp_to_eef[0], cp_to_eef[1],
                                                            eef_to_world[0], eef_to_world[1])

            # IK -> get human joint angles
            rightShoulder = 3
            rightElbow = 4
            q_H = self.bc_second.calculateInverseKinematics(self.humanoid._humanoid, rightElbow, targetPosition=cp_to_world[0], targetOrientation=cp_to_world[1])
            print('q_H:  ', q_H)

            # # FOR DEBUGGING
            # self.bc_second.setJointMotorControlMultiDof(self.humanoid._humanoid, rightShoulder, controlMode=p.POSITION_CONTROL, targetPosition=q_H[:3])
            # self.bc_second.setJointMotorControl2(self.humanoid._humanoid, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=q_H[3])
            # self.bc_second.stepSimulation()
            # cp_to_world = self.bc_second.getLinkState(self.humanoid._humanoid, rightElbow)[:2]

            # check q_H with joint limits, clamp
            clamped_q_H = []
            for i in range(len(q_H)):
                if q_H[i] < self.human_arm_lower_limits[i]:
                    print('violated lower', q_H[i], self.human_arm_lower_limits[i])
                    clamped_q_H.append(self.human_arm_lower_limits[i])
                elif q_H[i] > self.human_arm_upper_limits[i]:
                    print('violated upper', q_H[i], self.human_arm_upper_limits[i])
                    clamped_q_H.append(self.human_arm_upper_limits[i])
                else:
                    clamped_q_H.append(q_H[i])
            print('clamped_q_H:  ', clamped_q_H)

            # move humanoid in the 2nd server, get new cp pose
            self.bc_second.setJointMotorControlMultiDof(self.humanoid._humanoid, rightShoulder, controlMode=p.POSITION_CONTROL, targetPosition=clamped_q_H[:3])
            self.bc_second.setJointMotorControl2(self.humanoid._humanoid, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=clamped_q_H[3])
            self.bc_second.stepSimulation()
            cp_to_world = self.bc_second.getLinkState(self.humanoid._humanoid, rightElbow)[:2]

            # get new eef pose
            eef_to_world = self.bc_second.multiplyTransforms(self.eef_to_cp[0], self.eef_to_cp[1],
                                                            cp_to_world[0], cp_to_world[1])

            # IK -> get new robot joint angles
            q_R = self.bc_second.calculateInverseKinematics(self.robot.id, self.robot.eef_id, eef_to_world[0], eef_to_world[1],
                                                        self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                        maxNumIterations=20)
            q_R = [q_R[i] for i in range(len(self.robot.arm_controllable_joints))]
            q_R_list[idx] = torch.from_numpy(np.array(q_R)).double().to(device)
            print('q_R after clamping: ', q_R)
        
        # print('q_R_list after clamp: ', q_R_list)
        return q_R_list

    def violate_human_arm_limits(self, q_R):
        # get eef pose from q_R
        for i, joint in enumerate(self.robot.arm_controllable_joints):
            self.bc_second.setJointMotorControl2(self.robot.id, joint, p.POSITION_CONTROL, q_R[i])
            self.bc_second.stepSimulation()
        eef_to_world = self.bc_second.getLinkState(self.robot.id, self.robot.eef_id)[:2]

        # get cp pose
        cp_to_eef = self.bc_second.invertTransform(self.eef_to_cp[0], self.eef_to_cp[1])
        cp_to_world = self.bc_second.multiplyTransforms(cp_to_eef[0], cp_to_eef[1],
                                                        eef_to_world[0], eef_to_world[1])

        # IK -> get human joint angles
        right_shoulder_r = 3
        right_shoulder_p = 4
        right_shoulder_y = 5
        rightElbow = 7
        q_H = self.bc_second.calculateInverseKinematics(self.humanoid._humanoid, rightElbow, 
                                                        targetPosition=cp_to_world[0], targetOrientation=cp_to_world[1])
        print('q_H: ', q_H)
        print('lower limit: ', self.human_arm_lower_limits)
        print('upper limit: ', self.human_arm_upper_limits)

        # check q_H with joint limits
        for i in range(len(q_H)):
            if q_H[i] < self.human_arm_lower_limits[i]:
                return True
            elif q_H[i] > self.human_arm_upper_limits[i]:
                return True

        return False