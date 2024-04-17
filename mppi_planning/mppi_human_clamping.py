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
from humanoid_with_rev import Humanoid
from humanoid_with_rev import HumanoidPose

# ramp
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower

# Torch imports
import torch

class MPPI_H_Clamp():
    def __init__(self, eef_to_cp, robot_current_joint_angles, human_arm_lower_limits, human_arm_upper_limits, human_rest_poses):
        # 2nd BC server
        self.bc_second = BulletClient(connection_mode=p.DIRECT)
        self.bc_second.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc_second.configureDebugVisualizer(self.bc_second.COV_ENABLE_Y_AXIS_UP, 1)

        # load humanoid in the 2nd server
        motionPath = 'data/Sitting1.json'
        motion = MotionCaptureData()
        motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc_second, motion, [0, 0.3, 0])
        self.right_shoulder_y = 3
        self.right_shoulder_p = 4
        self.right_shoulder_r = 5
        self.right_elbow = 7
        # self.human_motion_from_frame_data(self.humanoid, motion, 60)

        # load robot in the 2nd server
        self.robot = UR5Robotiq85(self.bc_second, (-0.75, 0, 0), (-1.57, 0, 0), globalScaling=1.2)
        # self.robot = UR5Robotiq85(self.bc_second, (-0.75, 0.3, 0.25), (-1.57, 0, 0))
        self.robot.load()
        self.robot.reset()

        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            self.bc_second.resetJointState(self.robot.id, joint_id, robot_current_joint_angles[i])
        self.bc_second.stepSimulation()

        # initialize T_eef_cp (constant)
        self.eef_to_cp = eef_to_cp
        self.cp_to_eef = self.bc_second.invertTransform(self.eef_to_cp[0], self.eef_to_cp[1])

        # human arm joint parameters
        self.human_arm_lower_limits = human_arm_lower_limits
        self.human_arm_upper_limits = human_arm_upper_limits
        self.human_rest_poses = human_rest_poses
    
    def human_motion_from_frame_data(self, humanoid, motion, utNum):
        keyFrameDuration = motion.KeyFrameDuraction()
        self.bc_second.stepSimulation()
        humanoid.RenderReference(utNum * keyFrameDuration, self.bc_second)

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
            rp = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

            # IK -> get human joint angles
            q_H = self.bc_second.calculateInverseKinematics(self.humanoid._humanoid, self.right_elbow, 
                                                            targetPosition=world_to_cp[0], targetOrientation=world_to_cp[1],
                                                            lowerLimits=ll, upperLimits=ul,
                                                            jointRanges=jr, restPoses=rp,
                                                            maxNumIterations=100, residualThreshold=1e-6
                                                            )

            # move humanoid in the 2nd server, get new cp pose
            self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_y, q_H[0])
            self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_p, q_H[1])
            self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_r, q_H[2])
            self.bc_second.resetJointState(self.humanoid._humanoid, self.right_elbow, q_H[3])
            self.bc_second.stepSimulation()
            world_to_cp = self.bc_second.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]

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

    def violate_human_arm_limits(self, q_R):
        # get eef pose from q_R
        for i, joint in enumerate(self.robot.arm_controllable_joints):
            self.bc_second.resetJointState(self.robot.id, joint, q_R[i])
        self.bc_second.stepSimulation()
        eef_to_world = self.bc_second.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        # print("eef_to_world", eef_to_world)

        # get cp pose
        cp_to_eef = self.bc_second.invertTransform(self.eef_to_cp[0], self.eef_to_cp[1])
        cp_to_world = self.bc_second.multiplyTransforms(cp_to_eef[0], cp_to_eef[1],
                                                        eef_to_world[0], eef_to_world[1])

        

        # parameters for null space IK
        ll = self.human_arm_lower_limits
        ul = self.human_arm_upper_limits
        jr = list(np.array(ul) - np.array(ll))
        rp = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

        # IK -> get human joint angles
        q_H = self.bc_second.calculateInverseKinematics(self.humanoid._humanoid, self.right_elbow, 
                                                        targetPosition=cp_to_world[0], targetOrientation=cp_to_world[1],
                                                        lowerLimits=ll, upperLimits=ul,
                                                        jointRanges=jr, restPoses=rp,
                                                        maxNumIterations=100, residualThreshold=1e-6
                                                        )

        # check q_H with joint limits
        for i in range(len(q_H)):
            if q_H[i] < self.human_arm_lower_limits[i]:
                return True
            elif q_H[i] > self.human_arm_upper_limits[i]:
                return True
            
        # check if target cp pose is actually reachable with the computed IK sol
        threshold = 0.2
        self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_y, q_H[0])
        self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_p, q_H[1])
        self.bc_second.resetJointState(self.humanoid._humanoid, self.right_shoulder_r, q_H[2])
        self.bc_second.resetJointState(self.humanoid._humanoid, self.right_elbow, q_H[3])
        self.bc_second.stepSimulation()

        current_cp_to_world = self.bc_second.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        dist = np.linalg.norm(np.array(cp_to_world[0])-np.array(current_cp_to_world[0]))
        
        if (dist > threshold):
            return True

        return False