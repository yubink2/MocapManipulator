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
from scipy.spatial.transform import Rotation as R
from transformation import compose_matrix, inverse_matrix

# humanoid
from deep_mimic.env.motion_capture_data import MotionCaptureData
from humanoid_with_rev import Humanoid
from humanoid_with_rev import HumanoidPose

# ramp
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower

# mppi H clamp
from mppi_planning.mppi_human_clamping import MPPI_H_Clamp

# point cloud
import open3d as o3d
from utils.point_cloud_utils import *


# urdf paths
robot_urdf_location = 'pybullet_ur5/urdf/ur5_robotiq_85.urdf'
scene_urdf_location = 'resources/environment/environment.urdf'
control_points_location = 'resources/ur5_control_points/control_points.json'
control_points_number = 28

# UR5 parameters
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


class HumanDemo():
    def __init__(self):
        self.bc = BulletClient(connection_mode=p.GUI)
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_Y_AXIS_UP, 1)
        # self.bc.setGravity(0, -9.8, 0) 
        self.bc.setGravity(0, 0, -9.8) 
        self.bc.setTimestep = 0.0005
        # y2zOrn = self.bc.getQuaternionFromEuler((-1.57, 0, 0))

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, -0.04, 0))

        # add obstacle
        # sphere_pos1 = (1.5, 0.45, 0)
        # sphere_pos2 = (1.0, 0.45, 0)
        sphere_pos1 = (1.5, 0, 0.45)
        sphere_pos2 = (1.0, 0, 0.65)
        self.sphere1 = self.draw_sphere_marker(sphere_pos1, radius=0.05)
        self.sphere2 = self.draw_sphere_marker(sphere_pos2, radius=0.05)

        # get point cloud
        point_cloud = []
        point_cloud.extend(get_point_cloud_from_visual_shapes(self.sphere1))
        point_cloud.extend(get_point_cloud_from_visual_shapes(self.sphere2))
        self.pcd = np.array(point_cloud)

        # load robot
        robot_pose = ((1, 0, 0), (0, 0, 0))
        self.robot_base_to_world = compose_matrix(translate=robot_pose[0], angles=robot_pose[1])
        self.robot = UR5Robotiq85(self.bc, robot_pose[0], robot_pose[1])
        self.robot.load()
        self.robot.reset()

        # ###### DEBUGGING

        # q = [-0.4494, -0.9002,  1.2870, -1.1387, -1.4058,  0.3612]
        # control_points = [[ 1.0239,  0.0495,  0.0892],
        # [ 0.9761, -0.0495,  0.0892],
        # [ 1.0000,  0.0000,  0.0342],
        # [ 1.0829,  0.1719,  0.0892],
        # [ 1.0351,  0.0728,  0.0892],
        # [ 1.1501,  0.1395,  0.1832],
        # [ 1.1023,  0.0404,  0.1832],
        # [ 1.0157,  0.2043, -0.0049],
        # [ 0.9679,  0.1052, -0.0049],
        # [ 1.1771, -0.1341,  0.4410],
        # [ 1.2293, -0.0260,  0.4410],
        # [ 1.2606, -0.1744,  0.4033],
        # [ 1.3127, -0.0663,  0.4033],
        # [ 1.0854, -0.0899,  0.4825],
        # [ 1.1375,  0.0182,  0.4825],
        # [ 1.3523, -0.2186,  0.3618],
        # [ 1.4045, -0.1105,  0.3618],
        # [ 1.5261, -0.2358,  0.2263],
        # [ 1.6182, -0.2802,  0.3220],
        # [ 1.6152, -0.2788,  0.2230],
        # [ 1.6494, -0.1920,  0.2303],
        # [ 1.5756, -0.1564,  0.3180],
        # [ 1.5750, -0.2328,  0.2663],
        # [ 1.6476, -0.1679,  0.2484],
        # [ 1.6939, -0.2367,  0.1616],
        # [ 1.7139, -0.1617,  0.1958],
        # [ 1.5886, -0.1367,  0.1091],
        # [ 1.6579, -0.1920,  0.1900],
        # [ 1.6464, -0.1988,  0.1062],
        # [ 1.6001, -0.1300,  0.1929],
        # [ 1.6895, -0.1581,  0.0970],
        # [ 1.6432, -0.0894,  0.1837],
        # [ 1.7327, -0.1175,  0.0878],
        # [ 1.6864, -0.0487,  0.1746]]

        # self.move_robot(q)
        # for cp in control_points:
        #     self.draw_sphere_marker(position=cp, radius=0.03, color=[1,0,0,1])
        # print('done')

        # ###### DEBUGGING

        # compute start and goal config
        left = (1.65, -0.3, 0.4)
        right = (1.45, 0.4, 0.4)
        # left = (1.65, 0.4, -0.3)
        # right = (1.45, 0.4, 0.4)
        self.draw_sphere_marker(left, radius=0.05, color=[0,1,0,1])
        self.draw_sphere_marker(right, radius=0.05, color=[0,0,1,1])

        current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, left)
        self.current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
        target_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, right)
        self.target_joint_angles = [target_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
        
        self.move_robot(self.current_joint_angles)
        self.move_robot(self.target_joint_angles)
                      
    ### INITIALIZE PLANNER
    def init_traj_planner(self):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]
        
        # Instantiate trajectory planner
        self.trajectory_planner = TrajectoryPlanner(
            joint_limits=JOINT_LIMITS,
            robot_urdf_location=robot_urdf_location,
            scene_urdf_location=scene_urdf_location,
            link_fixed=LINK_FIXED,
            link_ee=LINK_EE,
            link_skeleton=LINK_SKELETON,
            control_points_location = control_points_location,
            control_points_number = control_points_number,
            mppi_H_clamp = None,
            world_to_robot_base = self.robot_base_to_world,
        )
        print("Instantiated trajectory planner")

    def init_mppi_planner(self, current_joint_angles, target_joint_angles, clamp_by_human):
        # MPPI parameters
        N_JOINTS = len(self.robot.arm_controllable_joints)
        mppi_control_limits = [
            -0.5 * np.ones(N_JOINTS),
            0.5 * np.ones(N_JOINTS)
        ]
        mppi_nsamples = 1000
        mppi_covariance = 0.05
        mppi_lambda = 1.0

        # Update current & target joint angles
        self.current_joint_angles = current_joint_angles
        self.target_joint_angles = target_joint_angles

        # Update whether to clamp_by_human
        self.clamp_by_human = clamp_by_human
        self.trajectory_planner.update_clamp_by_human(self.clamp_by_human)

        # Instantiate MPPI object
        self.trajectory_planner.instantiate_mppi_ja_to_ja(
            self.current_joint_angles,
            self.target_joint_angles,
            mppi_control_limits=mppi_control_limits,
            mppi_nsamples=mppi_nsamples,
            mppi_covariance=mppi_covariance,
            mppi_lambda=mppi_lambda,
            waypoint_density = 5,
            action_smoothing= 0.4,
        )
        print('Instantiated MPPI object')

        # Update environment point cloud
        self.trajectory_planner.update_obstacle_pcd(self.pcd)
        print("Updated environment point cloud")

        # Plan trajectory
        start_time = time.time()
        trajectory = self.trajectory_planner.get_mppi_rollout(self.current_joint_angles)
        print("planning time : ", time.time()-start_time)
        return trajectory
    
    def init_traj_follower(self):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]

        # Trajectory Follower initialization
        self.trajectory_follower = TrajectoryFollower(
            joint_limits = JOINT_LIMITS,
            robot_urdf_location = robot_urdf_location,
            control_points_json = control_points_location,
            link_fixed = LINK_FIXED,
            link_ee = LINK_EE,
            link_skeleton = LINK_SKELETON,
            control_points_number = control_points_number,
        )
        print('trajectory follower instantiated')

        # TODO update environment point cloud
        self.trajectory_follower.update_obstacle_pcd(self.pcd)
        print("Updated environment point cloud")
    
    def draw_sphere_marker(self, position, radius=0.07, color=[1, 0, 0, 1]):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
        return marker_id

    def move_robot(self, q_robot):
        # for _ in range(500):
        #     for i, joint_id in enumerate(self.robot.arm_controllable_joints):
        #         self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, q_robot[i],
        #                                         force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
        #     self.bc.stepSimulation()

        for _ in range(500):
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                self.bc.resetJointState(self.robot.id, joint_id, q_robot[i])
            self.bc.stepSimulation()
        


if __name__ == '__main__':
    env = HumanDemo()
    env.init_traj_planner()

    traj = env.init_mppi_planner(env.current_joint_angles, env.target_joint_angles, clamp_by_human=False)
    print('planner traj')
    print(traj)

    time.sleep(3)

    for q in traj:
        for _ in range (300):
            for i, joint_id in enumerate(env.robot.arm_controllable_joints):
                env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
                                            force=500)
            env.bc.stepSimulation()
        time.sleep(0.5)

    for i, joint_id in enumerate(env.robot.arm_controllable_joints):
        print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

    time.sleep(5)


    # ### TEST TRAJ FOLLOWER
    # env.init_traj_follower()
    # env.trajectory_follower.update_trajectory(traj)
    # velocity_command = env.trajectory_follower.follow_trajectory(env.current_joint_angles)
    # print('follower velocity command')
    # print(velocity_command)

    # while True:
    #     for q in velocity_command:
    #         env.move_robot(q)
    #         env.bc.stepSimulation()