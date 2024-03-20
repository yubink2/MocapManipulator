"""Credit to: https://github.com/lyfkyle/pybullet_ompl/"""

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

# humanoid
from deep_mimic.env.motion_capture_data import MotionCaptureData
from humanoid import Humanoid
from humanoid import HumanoidPose

# ramp
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower

# mppi H clamp
from mppi_planning.mppi_human_clamping_panda import MPPI_H_Clamp


class HumanDemo():
    def __init__(self):
        self.obstacles = []

        self.bc = BulletClient(connection_mode=p.GUI)
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_Y_AXIS_UP, 1)
        self.bc.setGravity(0, -9.8, 0) 
        self.bc.setTimestep = 0.0005

        y2zOrn = self.bc.getQuaternionFromEuler((-1.57, 0, 0))

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)
        bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed
        table1_id = self.bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.3), y2zOrn, globalScaling=0.6)  # table
        table2_id = self.bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.3), y2zOrn, globalScaling=0.6)  # table
        # block_id = self.bc.loadURDF("cube.urdf", (-1.0, 0.15, 0), y2zOrn, useFixedBase=True, globalScaling=0.45)  # block on robot

        # load human
        motionPath = 'data/Greeting.json'
        motion = MotionCaptureData()
        motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc, motion, [0, 0.3, 0])

        # add obstacles
        self.obstacles.append(plane_id)
        self.obstacles.append(bed_id)
        self.obstacles.append(table1_id)
        self.obstacles.append(table2_id)
        # self.obstacles.append(block_id)

        # load robot
        self.robotID = self.bc.loadURDF('resources/panda/panda.urdf', (-1.0, -0.04, 0), y2zOrn, useFixedBase=True)
        # self.robot_eef = 11
        self.robot_eef = 8
        self.init_robot_configs()

    def init_robot_configs(self):
        # open gripper
        open_length = 0.04
        for i in [9, 10]:
            self.bc.setJointMotorControl2(self.robotID, i, p.POSITION_CONTROL, open_length, force=20)
        for _ in range(50):
            self.bc.stepSimulation()

        # move robot to grasp pose
        eef_grasp_pose = ((-0.5196823277268179, 0.3667220578207198, 0.41692813317841715), 
                            (-0.06208896414048752, 0.821676027208712, -0.5664264377725382, -0.012432113045745374))
        target_joint_angles = self.bc.calculateInverseKinematics(self.robotID, self.robot_eef, eef_grasp_pose[0], eef_grasp_pose[1])[:7]
        # target_joint_angles = [-1.18, 0.45, 0.5, -1.3, 0.1, 1.5, 0]
        for i, joint_angle in enumerate(target_joint_angles):
            self.bc.setJointMotorControl2(self.robotID, i, p.POSITION_CONTROL, joint_angle)
        for _ in range(50):
            self.bc.stepSimulation()
        self.current_joint_angles = target_joint_angles

        # print robot info
        for i in range(self.bc.getNumJoints(self.robotID)):
            print(i, self.bc.getJointInfo(self.robotID, i))
            print(i, self.bc.getLinkState(self.robotID, i))
        
        # attach human arm (obj) to eef (body)
        body_pose = self.bc.getLinkState(self.robotID, self.robot_eef)  # eef to world
        obj_pose = self.bc.getLinkState(self.humanoid._humanoid, 4)  # cp to world
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])  # world to eef
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])  # world to eef * cp to world

        cid = self.bc.createConstraint(parentBodyUniqueId=self.robotID,
                            parentLinkIndex=self.robot_eef,
                            childBodyUniqueId=self.humanoid._humanoid,
                            childLinkIndex=4,
                            jointType=p.JOINT_FIXED,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))

    def draw_sphere_marker(self, position, radius, color):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        self.marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

    def init_mppi_planner(self):
        # urdf paths
        robot_urdf_location = 'resources/panda/panda.urdf'
        scene_urdf_location = 'resources/environment/environment.urdf'
        control_points_location = 'resources/panda_control_points/control_points.json'

        # panda parameters
        JOINT_LIMITS = [
            np.array([-2.8973, -1.7628, -2.8973, -
                        -3.0718, -2.8973, -0.0175, -2.8973]),
            np.array([2.8973, 1.7628, 2.8973, -
                        -0.0698, 2.8973, 3.7525, 2.8973])
        ]

        LINK_FIXED = 'panda_link0'
        LINK_EE = 'panda_hand'

        LINK_SKELETON = [
            'panda_link1',
            'panda_link3',
            'panda_link4',
            'panda_link5',
            'panda_link7',
            'panda_hand',
        ]

        # Instantiate mppi H clamp
        eef_to_world = self.bc.getLinkState(self.robotID, self.robot_eef)[:2]
        cp_to_world = self.bc.getLinkState(self.humanoid._humanoid, 4)[:2]
        world_to_cp = self.bc.invertTransform(cp_to_world[0], cp_to_world[1])
        eef_to_cp = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                        world_to_cp[0], world_to_cp[1])
        print('*** eef_to_cp', eef_to_cp)

        shoulder_min = env.bc.getEulerFromQuaternion([-0.397618, -0.885139, -0.759182, -0.014004])
        shoulder_max = env.bc.getEulerFromQuaternion([0.654265, 0.959910, 0.718185, 0.643939])
        elbow_min = [0.4]
        elbow_max = [2.5]
        human_arm_lower_limits = shoulder_min + shoulder_max
        human_arm_upper_limits = elbow_min + elbow_max

        self.mppi_H_clamp = MPPI_H_Clamp(eef_to_cp, human_arm_lower_limits, human_arm_upper_limits, self.current_joint_angles)

        # Instantiate trajectory planner
        self.trajectory_planner = TrajectoryPlanner(
            joint_limits=JOINT_LIMITS,
            robot_urdf_location=robot_urdf_location,
            scene_urdf_location=scene_urdf_location,
            link_fixed=LINK_FIXED,
            link_ee=LINK_EE,
            link_skeleton=LINK_SKELETON,
            control_points_location = control_points_location,
            mppi_H_clamp = self.mppi_H_clamp,
        )
        print("Instantiated trajectory planner")

        # # Trajectory Follower initialization
        # self.trajectory_follower = TrajectoryFollower(
        #     joint_limits = JOINT_LIMITS,
        #     robot_urdf_location = robot_urdf_location,
        #     link_fixed = LINK_FIXED,
        #     link_ee = LINK_EE,
        #     link_skeleton = LINK_SKELETON,
        # )
        # print('trajectory follower instantiated')

        # MPPI parameters
        N_JOINTS = 7
        mppi_control_limits = [
            -0.05 * np.ones(N_JOINTS),
            0.05 * np.ones(N_JOINTS)
        ]
        mppi_nsamples = 500
        mppi_covariance = 0.005
        mppi_lambda = 1.0

        # human goal config
        cp_pos, cp_orn = env.bc.getLinkState(env.humanoid._humanoid, 4)[:2]  
        cp_pos_up = (cp_pos[0]-0.15, cp_pos[1]+0.4, cp_pos[2]+0.15)
        cp_orn = env.bc.getQuaternionFromEuler((-1.57, 0.15, -1.57))
        cp_orn = (-0.06208896414048752, 0.821676027208712, -0.5664264377725382, -0.012432113045745374)

        # mark goal pose
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
        marker_id = self.bc.createMultiBody(basePosition=cp_pos_up, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

        # Find joint angles
        current_joint_angles = self.current_joint_angles
        # target_joint_angles = self.bc.calculateInverseKinematics(self.robotID, self.robot_eef, cp_pos_up, cp_orn)[:7]
        target_joint_angles = self.bc.calculateInverseKinematics(self.robotID, self.robot_eef, cp_pos_up)[:7]
        print('goal: ', target_joint_angles)

        # Instantiate MPPI object
        self.trajectory_planner.instantiate_mppi_ja_to_ja(
            current_joint_angles,
            target_joint_angles,
            mppi_control_limits=mppi_control_limits,
            mppi_nsamples=mppi_nsamples,
            mppi_covariance=mppi_covariance,
            mppi_lambda=mppi_lambda,
        )
        print('Instantiate MPPI object')

        # Plan trajectory
        trajectory = self.trajectory_planner.get_mppi_rollout(current_joint_angles)
        return trajectory


if __name__ == '__main__':
    env = HumanDemo()

    traj = env.init_mppi_planner()
    time.sleep(2)

    print('traj: ', traj)
    for q in traj:
        for i, joint_angle in enumerate(q):
            env.bc.setJointMotorControl2(env.robotID, i, p.POSITION_CONTROL, joint_angle)
        env.bc.stepSimulation() 
        time.sleep(0.1)

    # joint_11 = env.bc.getLinkState(env.robotID, 11)[:2]
    joint_8 = env.bc.getLinkState(env.robotID, 8)[:2]
    vs_id = env.bc.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
    marker_id = env.bc.createMultiBody(basePosition=joint_8[0], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

    while(True):
        time.sleep(0.1)

    env.bc.disconnect()

