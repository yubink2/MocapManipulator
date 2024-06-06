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

# ramp
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower

# Torch imports
import torch

class MPPI_H_Clamp():
    def __init__(self, eef_to_cp, human_arm_lower_limits, human_arm_upper_limits, human_rest_poses):
        # 2nd BC server
        self.bc_second = BulletClient(connection_mode=p.DIRECT)
        self.bc_second.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc_second.configureDebugVisualizer(self.bc_second.COV_ENABLE_Y_AXIS_UP, 1)

        # load humanoid in the 2nd server
        baseOrn = self.bc_second.getQuaternionFromEuler((1.57, 0, 1.57))
        self.humanoid = self.bc_second.loadURDF("urdf/humanoid_for_rwe.urdf", (0, 0.08, 0), baseOrn, 
                                                globalScaling=0.33, useFixedBase=True)
        self.left_shoulder = 6
        self.left_elbow = 7

        # load robot in the 2nd server
        self.robot = UR5Robotiq85(self.bc_second, (-0.95, 0.53, 0.9), (-1.57, 0, 0), globalScaling=1.0)
        self.robot.load()
        self.robot.reset()
        self.robot.open_gripper()

        # initialize T_eef_cp (constant)
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = self.bc_second.invertTransform(self.eef_to_cp[0], self.eef_to_cp[1])

        # human arm joint parameters
        self.human_arm_lower_limits = human_arm_lower_limits
        self.human_arm_upper_limits = human_arm_upper_limits
        self.human_rest_poses = human_rest_poses
    
    def clamp_human_joints(self, q_R_list, device):
        for idx, q_R in enumerate(q_R_list):
            # get eef pose from q_R
            for i, joint in enumerate(self.robot.arm_controllable_joints):
                self.bc_second.resetJointState(self.robot.id, joint, q_R[i])
            self.bc_second.stepSimulation()
            world_to_eef = self.bc_second.getLinkState(self.robot.id, self.robot.eef_id)[:2]

            # get cp pose
            world_to_cp = self.bc_second.multiplyTransforms(world_to_eef[0], world_to_eef[1],
                                                            self.eef_to_cp[0], self.eef_to_cp[1])

            # human parameters for null space IK
            ll = self.human_arm_lower_limits
            ul = self.human_arm_upper_limits
            jr = list(np.array(ul) - np.array(ll))
            rp = self.human_rest_poses

            # IK -> get human joint angles
            q_H = self.bc_second.calculateInverseKinematics(self.humanoid, self.left_elbow, 
                                                            targetPosition=world_to_cp[0], targetOrientation=world_to_cp[1],
                                                            lowerLimits=ll, upperLimits=ul,
                                                            jointRanges=jr, restPoses=rp,
                                                            maxNumIterations=100, residualThreshold=1e-6
                                                            )

            # move humanoid in the 2nd server, get new cp pose
            self.bc_second.resetJointState(self.humanoid, self.left_shoulder, q_H[0])
            self.bc_second.resetJointState(self.humanoid, self.left_elbow, q_H[1])
            self.bc_second.stepSimulation()
            world_to_cp = self.bc_second.getLinkState(self.humanoid, self.left_elbow)[:2]

            # get new eef pose
            world_to_eef = self.bc_second.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                            self.cp_to_eef[0], self.cp_to_eef[1])

            # IK -> get new robot joint angles
            q_R = self.bc_second.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                        self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                        maxNumIterations=20)
            q_R = [q_R[i] for i in range(len(self.robot.arm_controllable_joints))]
            q_R_list[idx] = torch.from_numpy(np.array(q_R)).double().to(device)
        
        return q_R_list
    
    