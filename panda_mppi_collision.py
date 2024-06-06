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

# ramp
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower

# mppi H clamp
from mppi_planning.mppi_human_clamping_panda import MPPI_H_Clamp

# point cloud
import open3d as o3d
from utils.point_cloud_utils import *


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


class HumanDemo():
    def __init__(self):
        self.bc = BulletClient(connection_mode=p.GUI)
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_Y_AXIS_UP, 1)
        self.bc.setGravity(0, 0, -9.8) 
        self.bc.setTimestep = 0.0005

        # y2zOrn = self.bc.getQuaternionFromEuler((-1.57, 0, 0))

        # load environment
        # plane_id = self.bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)
        plane_id = self.bc.loadURDF("plane.urdf", (0, 0, -0.04))

        # add sphere
        # sphere_pos1 = (0.5, 0.4, 1.0)   # above goal
        # sphere_pos2 = (-0.2, 0, 1.0)   # above start
        # sphere_pos3 = (0.75, -0.1, 1.0)   # above start
        # self.sphere1 = self.draw_sphere_marker(sphere_pos1, radius=0.3, color=[1,0,0,1])
        # self.sphere2 = self.draw_sphere_marker(sphere_pos2, radius=0.3, color=[1,0,0,1])
        # self.sphere3 = self.draw_sphere_marker(sphere_pos3, radius=0.4, color=[1,0,0,1])

        # load robot
        self.robotID = self.bc.loadURDF('resources/panda/panda.urdf', (0, 0, 0), useFixedBase=True)
        # self.robotID = self.bc.loadURDF('resources/panda/panda.urdf', (0, 0, 0), y2zOrn, useFixedBase=True)
        self.robot_eef = 8
        self.reset_robot()

        # q = [ 0.6104,  0.5091, -0.0192,  0.0698, -0.2888,  2.8111,  2.3200]
        # self.move_robot(q)

        # add obstacle
        sphere_pos = (0.5, 0.4, 0.5)
        self.sphere = self.draw_sphere_marker(sphere_pos, radius=0.1, color=[1,0,0,1])

        # current and target joint angles
        left = (0.65, 0, 0.5)
        right = (0.65, 0, 1.0)
        # right = (0.35, 0.7, 0.5)
        self.draw_sphere_marker(left, radius=0.05, color=[0,1,0,1])   # green
        self.draw_sphere_marker(right, radius=0.05, color=[0,0,1,1])  # blue

        # get point cloud
        point_cloud = []
        # point_cloud.extend(get_point_cloud_from_collision_shapes(self.sphere1))
        # point_cloud.extend(get_point_cloud_from_collision_shapes(self.sphere2))
        # point_cloud.extend(get_point_cloud_from_collision_shapes(self.sphere3))
        point_cloud.extend(get_point_cloud_from_collision_shapes(self.sphere))
        self.pcd = np.array(point_cloud)

        self.current_joint_angles = self.bc.calculateInverseKinematics(self.robotID, self.robot_eef, left)[:7]
        self.move_robot(self.current_joint_angles)

        self.target_joint_angles = self.bc.calculateInverseKinematics(self.robotID, self.robot_eef, right)[:7]
        # self.move_robot(self.target_joint_angles)

    def draw_sphere_marker(self, position, radius, color):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        # col_id = self.bc.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
        return marker_id

    def init_mppi_planner(self):
        # Instantiate trajectory planner
        self.trajectory_planner = TrajectoryPlanner(
            joint_limits=JOINT_LIMITS,
            robot_urdf_location=robot_urdf_location,
            scene_urdf_location=scene_urdf_location,
            link_fixed=LINK_FIXED,
            link_ee=LINK_EE,
            link_skeleton=LINK_SKELETON,
            control_points_location = control_points_location,
            mppi_H_clamp = None,
        )
        print("Instantiated trajectory planner")

        self.trajectory_planner.update_clamp_by_human(clamp_by_human=False)

        # Update point cloud
        self.trajectory_planner.update_obstacle_pcd(self.pcd)
        print("Updated environment point cloud")

        # MPPI parameters
        N_JOINTS = 7
        mppi_control_limits = [
            -0.05 * np.ones(N_JOINTS),
            0.05 * np.ones(N_JOINTS)
        ]
        mppi_nsamples = 500
        mppi_covariance = 0.005
        mppi_lambda = 1.0

        # Instantiate MPPI object
        self.trajectory_planner.instantiate_mppi_ja_to_ja(
            self.current_joint_angles,
            self.target_joint_angles,
            mppi_control_limits=mppi_control_limits,
            mppi_nsamples=mppi_nsamples,
            mppi_covariance=mppi_covariance,
            mppi_lambda=mppi_lambda,
        )
        print('Instantiate MPPI object')

        # Plan trajectory
        trajectory = self.trajectory_planner.get_mppi_rollout(self.current_joint_angles)
        return trajectory
    
    def init_traj_follower(self):
        # Trajectory Follower initialization
        self.trajectory_follower = TrajectoryFollower(
            joint_limits = JOINT_LIMITS,
            robot_urdf_location = robot_urdf_location,
            control_points_json = control_points_location,
            link_fixed = LINK_FIXED,
            link_ee = LINK_EE,
            link_skeleton = LINK_SKELETON,
            # control_points_number = 31,
        )
        print('trajectory follower instantiated')

        # TODO update environment point cloud
        self.trajectory_follower.update_obstacle_pcd(self.pcd)
        print("Updated environment point cloud")

    
    def reset_robot(self):
        arm_rest_poses = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32]
        for _ in range (300):
            for i, joint in enumerate(arm_rest_poses):
                self.bc.setJointMotorControl2(self.robotID, i, p.POSITION_CONTROL, joint)
            self.bc.stepSimulation() 
    
    def move_robot(self, q):
        for _ in range (300):
            for i, joint in enumerate(q):
                self.bc.setJointMotorControl2(self.robotID, i, p.POSITION_CONTROL, joint)
            self.bc.stepSimulation() 

    def get_robot_config(self):
        config = []
        for joint_id in range(self.bc.getNumJoints(self.robotID)):
            config.append(self.bc.getJointState(self.robotID, joint_id)[0])
        return config[:7]



if __name__ == '__main__':
    env = HumanDemo()

    ###
    q = [ 0.6104,  0.5091, -0.0192,  0.0698, -0.2888,  2.8111,  2.3200]
    control_points = [[ 0.0742, -0.1060,  0.3330],
        [-0.0742,  0.1060,  0.3330],
        [ 0.0459,  0.0993,  0.5024],
        [ 0.1090,  0.0091,  0.5024],
        [ 0.1168,  0.0817,  0.4756],
        [ 0.0381,  0.0267,  0.5292],
        [ 0.0944,  0.0668,  0.6049],
        [ 0.1240,  0.2203,  0.5677],
        [ 0.2480,  0.0366,  0.5698],
        [ 0.1074,  0.0759,  0.6377],
        [ 0.1258,  0.0895,  0.7394],
        [ 0.2085,  0.1448,  0.6927],
        [ 0.1976,  0.0720,  0.7165],
        [ 0.1361,  0.1631,  0.7155],
        [ 0.2289,  0.3030,  0.9632],
        [ 0.1983,  0.2685,  0.8758],
        [ 0.1622,  0.2087,  0.7610],
        [ 0.2017,  0.1214,  0.7937],
        [ 0.2308,  0.1663,  0.9128],
        [ 0.2920,  0.1881,  0.9190],
        [ 0.2870,  0.1957,  0.9861],
        [ 0.2719,  0.1272,  0.9622],
        [ 0.1459,  0.1429,  0.9545],
        [ 0.1523,  0.1553,  1.0280],
        [ 0.3177,  0.1770,  1.0196],
        [ 0.3008,  0.2311,  1.0108],
        [ 0.2336,  0.2353,  1.0659],
        [ 0.2596,  0.1524,  1.0795],
        [ 0.2144,  0.1886,  1.1023],
        [ 0.3693,  0.2299,  1.0628],
        [ 0.2249,  0.2069,  1.1989]]

    
    # env.move_robot(q)
    # print(env.get_robot_config())

    # for cp in control_points:
    #     env.draw_sphere_marker(position=cp, radius=0.03, color=[1,0,0,1])

    # print('done')

    q_list = [
        [ 0.6104,  0.5091, -0.0192,  0.0698, -0.2888,  2.8111,  2.3200],
        [ 0.2885,  0.2861, -0.4121,  0.0698, -0.3560,  2.7946,  2.3219],
        [ 0.5059,  0.2740, -0.2357,  0.0698, -0.1940,  2.7694,  2.3843],
        [ 0.2604,  0.4361, -0.3692,  0.0698, -0.3067,  2.7608,  2.2700],
        [ 0.4595,  0.5818, -0.1120,  0.0698, -0.4471,  2.9354,  2.2830],
        [ 0.4104,  0.4335, -0.2880,  0.0698, -0.3202,  2.7081,  2.4540],
        [ 0.4736,  0.5734, -0.1319,  0.0698, -0.3640,  2.8471,  2.5041]
    ]
    for q in q_list:
        env.move_robot(q)

    ###
    traj = env.init_mppi_planner()
    time.sleep(10)

    # ###
    # env.init_traj_follower()
    # env.trajectory_follower.update_trajectory(traj)
    # velocity_command = env.trajectory_follower.follow_trajectory(env.current_joint_angles)
    # print('follower velocity command')
    # print(velocity_command)

    # while True:
    #     for q in velocity_command:
    #         env.move_robot(q)
    #         env.bc.stepSimulation()

    ###
    print('traj: ', traj)
    for q in traj:
        for _ in range (300):
            for i, joint_angle in enumerate(q):
                env.bc.setJointMotorControl2(env.robotID, i, p.POSITION_CONTROL, joint_angle)
            env.bc.stepSimulation() 
        time.sleep(0.1)

    while(True):
        time.sleep(0.1)

    env.bc.disconnect()

