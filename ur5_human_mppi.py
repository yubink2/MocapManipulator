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
from mppi_planning.mppi_human_clamping import MPPI_H_Clamp

# informed rrt star
from informed_rrtstar.informed_rrtstar_3d import InformedRRTStar


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
        block_id = self.bc.loadURDF("cube.urdf", (-1.0, 0.15, 0), y2zOrn, useFixedBase=True, globalScaling=0.45)  # block on robot

        # load human
        motionPath = 'data/Greeting.json'
        motion = MotionCaptureData()
        motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc, motion, [0, 0.3, 0])

        # while(True):
        #     self.bc.stepSimulation()

        # add obstacles
        self.obstacles.append(plane_id)
        self.obstacles.append(bed_id)
        self.obstacles.append(table1_id)
        self.obstacles.append(table2_id)
        # self.obstacles.append(block_id)

        # load robot
        self.robot = UR5Robotiq85(self.bc, (-1.0, 0.35, 0), (-1.57, 0, 0))
        # self.robot = UR5Robotiq85(self.bc, (-1.0, 0.2, 0), (-1.57, 0, 0))
        self.robot.load()
        self.robot.reset()
        self.init_robot_configs()

    def init_robot_configs(self):
        self.current_joint_angles = [self.robot.arm_rest_poses[i] for i in range(len(self.robot.arm_controllable_joints))]

        # # move robot to grasp pose
        # pos, orn = self.bc.getLinkState(self.humanoid._humanoid, 4)[:2]  
        # pos_up = (pos[0], pos[1]+0.13, pos[2])
        # orn = self.bc.getQuaternionFromEuler((-1.57, 0.15, -1.57))

        # current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, pos_up, orn,
        #                                                self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
        #                                                maxNumIterations=20)
        # self.current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]

        # for _ in range (50):
        #     for i, joint_id in enumerate(self.robot.arm_controllable_joints):
        #         self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, current_joint_angles[i],
        #                                         force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
        #         self.bc.stepSimulation()
        #     # time.sleep(0.1)
        # print('moved robot to init config')

        # # attach human arm (obj) to eef (body)
        # body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)  # eef to world
        # obj_pose = self.bc.getLinkState(self.humanoid._humanoid, 4)  # cp to world
        # world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])  # world to eef
        # obj_to_body = self.bc.multiplyTransforms(world_to_body[0],
        #                                     world_to_body[1],
        #                                     obj_pose[0], obj_pose[1])  # world to eef * cp to world

        # cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
        #                     parentLinkIndex=self.robot.eef_id,
        #                     childBodyUniqueId=self.humanoid._humanoid,
        #                     childLinkIndex=4,
        #                     jointType=p.JOINT_FIXED,
        #                     jointAxis=(0, 0, 0),
        #                     parentFramePosition=obj_to_body[0],
        #                     parentFrameOrientation=obj_to_body[1],
        #                     childFramePosition=(0, 0, 0),
        #                     childFrameOrientation=(0, 0, 0))

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def draw_sphere_marker(self, position, radius=0.07, color=[1, 0, 0, 1]):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        self.marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

    def init_mppi_planner(self):
        # urdf paths
        robot_urdf_location = 'pybullet_ur5/urdf/ur5_robotiq_85.urdf'
        scene_urdf_location = 'resources/environment/environment.urdf'
        control_points_location = 'resources/ur5_control_points/control_points.json'

        # UR5 parameters
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]
        LINK_FIXED = 'base_link'
        LINK_EE = 'ee_link'
        LINK_SKELETON = [
            'shoulder_link',
            'upper_arm_link',
            'forearm_link',
            'wrist_1_link',
            'wrist_2_link',
            'wrist_3_link',
            'ee_link',
        ]

        # Instantiate mppi H clamp
        eef_to_world = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        cp_to_world = self.bc.getLinkState(self.humanoid._humanoid, 4)[:2]
        world_to_cp = self.bc.invertTransform(cp_to_world[0], cp_to_world[1])
        eef_to_cp = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                        world_to_cp[0], world_to_cp[1])

        shoulder_min = [-2.583238756496965, -0.248997453133789, -3.1402077384521765]
        shoulder_max = [-1.3229245882839409, 1.2392816988875348, 3.1415394736319917]
        # elbow_min = [0.401146]
        elbow_min = [0]
        elbow_max = [2.541304]  
        self.human_arm_lower_limits = shoulder_min + elbow_min
        self.human_arm_upper_limits = shoulder_max + elbow_max

        self.mppi_H_clamp = MPPI_H_Clamp(eef_to_cp, self.human_arm_lower_limits, self.human_arm_upper_limits, self.current_joint_angles)

        # Instantiate trajectory planner
        self.trajectory_planner = TrajectoryPlanner(
            joint_limits=JOINT_LIMITS,
            robot_urdf_location=robot_urdf_location,
            scene_urdf_location=scene_urdf_location,
            link_fixed=LINK_FIXED,
            link_ee=LINK_EE,
            link_skeleton=LINK_SKELETON,
            control_points_location = control_points_location,
            control_points_number = 21,
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
        N_JOINTS = len(self.robot.arm_controllable_joints)
        mppi_control_limits = [
            -0.05 * np.ones(N_JOINTS),
            0.05 * np.ones(N_JOINTS)
        ]
        mppi_nsamples = 500
        mppi_covariance = 0.005
        mppi_lambda = 1.0

        # human goal config
        cp_pos, cp_orn = env.bc.getLinkState(env.humanoid._humanoid, 4)[:2]  
        cp_pos_up = (cp_pos[0]+0.2, cp_pos[1]+0.5, cp_pos[2]-0.5)
        cp_orn = env.bc.getQuaternionFromEuler((-1.57, 0.15, -1.57))

        # mark goal pose
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
        marker_id = self.bc.createMultiBody(basePosition=cp_pos_up, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

        # Find joint angles
        current_joint_angles = self.current_joint_angles
        target_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, cp_pos_up, cp_orn,
                                            self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                            maxNumIterations=20)
        target_joint_angles = [target_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
        self.target_joint_angles = target_joint_angles
        print("q_init: ", current_joint_angles)
        print("q_goal: ", target_joint_angles)

        # for _ in range(500):
        #     # self.robot.move_ee(action=current_joint_angles, control_method="joint")
        #     for i, joint_id in enumerate(self.robot.arm_controllable_joints):
        #         self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, target_joint_angles[i],
        #                                       force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
        #     self.bc.stepSimulation() 
        #     time.sleep(0.1)
        # sys.exit()

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

        # # Plan trajectory
        # start_time = time.time()
        # trajectory = self.trajectory_planner.get_mppi_rollout(current_joint_angles)
        # print("planning time : ", time.time()-start_time)
        # return trajectory

        return []
    
    def init_informed_rrtstar_planner(self, traj):
        self.informed_rrtstar_planner = InformedRRTStar(self.current_joint_angles, self.target_joint_angles, self.obstacles,
                                                        self.robot.id, self.robot.arm_controllable_joints)
        return self.informed_rrtstar_planner.plan(traj)
    
    def test_clamping(self, q_R):
        rightShoulder = 3
        rightElbow = 4
        current_shoulder = self.bc.getJointStateMultiDof(self.humanoid._humanoid, rightShoulder)[0]
        current_shoulder = self.bc.getEulerFromQuaternion(current_shoulder)
        current_elbow = self.bc.getJointState(self.humanoid._humanoid, rightElbow)[0]
        q_H = list(current_shoulder) + [current_elbow]
        print('current q_H: ', q_H)

        cp_pos, cp_orn = self.bc.getLinkState(self.humanoid._humanoid, rightElbow)[:2]
        
        #lower limits for null space
        ll = self.human_arm_lower_limits
        #upper limits for null space
        ul = self.human_arm_upper_limits
        #joint ranges for null space
        jr = list(np.array(ul) - np.array(ll))
        print('joint ranges:', jr)
        #restposes for null space
        rp = q_H

        q_H = self.bc.calculateInverseKinematics(self.humanoid._humanoid, rightElbow, 
                                                targetPosition=cp_pos, targetOrientation=cp_orn,
                                                lowerLimits=ll, upperLimits=ul,
                                                jointRanges=jr, restPoses=rp,
                                                maxNumIterations=300, residualThreshold=1e-6)
        print('calculated q_H from IK: ', q_H)
        target_shoulder = self.bc.getQuaternionFromEuler(q_H[:3])
        print('target shoulder:', target_shoulder)
        time.sleep(2)

        for _ in range(5000):
            self.bc.setJointMotorControlMultiDof(self.humanoid._humanoid, rightShoulder, controlMode=p.POSITION_CONTROL, targetPosition=target_shoulder)
            self.bc.setJointMotorControl2(self.humanoid._humanoid, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=q_H[3])
            self.bc.stepSimulation()

        current_shoulder = self.bc.getJointStateMultiDof(self.humanoid._humanoid, rightShoulder)[0]
        current_shoulder = self.bc.getEulerFromQuaternion(current_shoulder)
        current_elbow = self.bc.getJointState(self.humanoid._humanoid, rightElbow)[0]
        q_H = list(current_shoulder) + [current_elbow]
        print('current q_H after: ', q_H)

        # if env.mppi_H_clamp.violate_human_arm_limits(q_R):
        #     print('violated')

    def test_clamping2(self, q_R):
        for i in range(self.bc.getNumJoints(self.humanoid._humanoid)):
            print(self.bc.getJointInfo(self.humanoid._humanoid, i))
        right_shoulder_r = 3
        right_shoulder_p = 4
        right_shoulder_y = 5
        rightElbow = 7

        # q_H_test = [-1.8193453925837375, -0.04536734052020464, 2.478893589137632, 0.45577464302391746]
        # q_H_test = [3.14, 0.0, 3.14, 0.455]
        q_H_test = [-1.3229245882839409, 1.2392816988875348, 3.1415394736319917, 2.541304]
        # for _ in range(300):
        while(True):
            self.bc.setJointMotorControl2(self.humanoid._humanoid, right_shoulder_r, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[0])
            self.bc.setJointMotorControl2(self.humanoid._humanoid, right_shoulder_p, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[1])
            self.bc.setJointMotorControl2(self.humanoid._humanoid, right_shoulder_y, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[2])
            self.bc.setJointMotorControl2(self.humanoid._humanoid, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[3])
            self.bc.stepSimulation()

        current_shoulder = []
        current_shoulder.append(self.bc.getJointState(self.humanoid._humanoid, right_shoulder_r)[0])
        current_shoulder.append(self.bc.getJointState(self.humanoid._humanoid, right_shoulder_p)[0])
        current_shoulder.append(self.bc.getJointState(self.humanoid._humanoid, right_shoulder_y)[0])
        current_elbow = self.bc.getJointState(self.humanoid._humanoid, rightElbow)[0]
        q_H = list(current_shoulder) + [current_elbow]
        print('current q_H: ', q_H)

        cp_pos, cp_orn = self.bc.getLinkState(self.humanoid._humanoid, rightElbow)[:2]
        self.draw_sphere_marker(cp_pos)
        sys.exit()
        
        #lower limits for null space
        ll = self.human_arm_lower_limits
        #upper limits for null space
        ul = self.human_arm_upper_limits
        #joint ranges for null space
        jr = list(np.array(ul) - np.array(ll))
        #restposes for null space
        rp = q_H

        q_H = self.bc.calculateInverseKinematics(self.humanoid._humanoid, rightElbow, 
                                                targetPosition=cp_pos, targetOrientation=cp_orn,
                                                # lowerLimits=ll, upperLimits=ul,
                                                # jointRanges=jr, restPoses=rp,
                                                # maxNumIterations=100, residualThreshold=0.0001
                                                )
        print('calculated q_H from IK: ', q_H)
        time.sleep(2)

        for _ in range(50):
            self.bc.setJointMotorControl2(self.humanoid._humanoid, right_shoulder_r, controlMode=p.POSITION_CONTROL, targetPosition=q_H[0])
            self.bc.setJointMotorControl2(self.humanoid._humanoid, right_shoulder_p, controlMode=p.POSITION_CONTROL, targetPosition=q_H[1])
            self.bc.setJointMotorControl2(self.humanoid._humanoid, right_shoulder_y, controlMode=p.POSITION_CONTROL, targetPosition=q_H[2])
            self.bc.setJointMotorControl2(self.humanoid._humanoid, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=q_H[3])
            self.bc.stepSimulation()

        current_shoulder = []
        current_shoulder.append(self.bc.getJointState(self.humanoid._humanoid, right_shoulder_r)[0])
        current_shoulder.append(self.bc.getJointState(self.humanoid._humanoid, right_shoulder_p)[0])
        current_shoulder.append(self.bc.getJointState(self.humanoid._humanoid, right_shoulder_y)[0])
        current_elbow = self.bc.getJointState(self.humanoid._humanoid, rightElbow)[0]
        q_H = list(current_shoulder) + [current_elbow]
        print('current q_H after: ', q_H)

        if env.mppi_H_clamp.violate_human_arm_limits(q_R):
            print('violated')
        


if __name__ == '__main__':
    env = HumanDemo()

    traj = env.init_mppi_planner()
    print('MPPI planner done')

    q_R = [-0.96537332, -1.05807295,  1.56956323, -1.9578728,  -1.48554134, -0.96998654]
    env.test_clamping(q_R)

    # print('traj before:', traj)

    # for q in traj:
    #     if env.mppi_H_clamp.violate_human_arm_limits(q):
    #         print(f'removed {q}')
    #         traj.remove(q)

    # print('traj after:', traj)

    # flag = False
    # for _ in range(100):
    #     # env.bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
    #     if not flag:
    #         for q in traj:
    #             for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #                 env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
    #                                             force=env.robot.joints[joint_id].maxForce, maxVelocity=env.robot.joints[joint_id].maxVelocity)
    #             env.bc.stepSimulation() 
    #         flag = True
    #     time.sleep(0.1)
    #     env.bc.stepSimulation()

    # eef_pose = env.bc.getLinkState(env.robot.id, env.robot.eef_id)[:2]
    # vs_id = env.bc.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
    # marker_id = env.bc.createMultiBody(basePosition=eef_pose[0], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)


    # # traj2 = env.init_informed_rrtstar_planner(traj[:len(traj)-1])
    # traj2 = env.init_informed_rrtstar_planner(traj)
    # print('informed rrtstar planner done')
    # print(traj2)

    # time.sleep(2)
    # env.bc.disconnect()

