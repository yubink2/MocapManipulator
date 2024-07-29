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
from utils.debug_utils import *
from utils.transform_utils import *

# humanoid
from deep_mimic.env.motion_capture_data import MotionCaptureData
from humanoid_with_rev_xyz import Humanoid
from humanoid_with_rev_xyz import HumanoidPose

# ramp
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower

# mppi H clamp
from mppi_planning.mppi_human_clamping import MPPI_H_Clamp

# point cloud
import open3d as o3d
from utils.point_cloud_utils import *

# # contact graspnet
from contact_graspnet_pytorch.contact_graspnet_pytorch.inference import inference
from contact_graspnet_pytorch.contact_graspnet_pytorch.config_utils import load_config
from utils.grasp_utils import *

# informed rrt star
from informed_rrtstar.informed_rrtstar_3d import InformedRRTStar
from utils.collision_utils import get_collision_fn


# urdf paths
robot_urdf_location = 'pybullet_ur5/urdf/ur5_robotiq_85.urdf'
scene_urdf_location = 'resources/environment/environment.urdf'
# control_points_location = 'resources/ur5_control_points/control_points.json'
# control_points_number = 28
control_points_location = 'resources/ur5_control_points/T_control_points.json'
control_points_number = 55

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
        self.bc.setGravity(0, 0, -9.8) 
        self.bc.setTimestep = 0.05

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, -0.04, 0))
        self.bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, -0.1, 0.0), useFixedBase=True)  # bed
        self.cid = None

        # load human
        human_base_pos = (0, 0, 0.3)
        human_base_orn = self.bc.getQuaternionFromEuler((0, 1.57, 0))
        motionPath = 'data/Sitting1.json'
        self.motion = MotionCaptureData()
        self.motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc, self.motion, baseShift=human_base_pos, ornShift=human_base_orn)
        self.right_shoulder_y = 3
        self.right_shoulder_p = 4
        self.right_shoulder_r = 5
        self.right_shoulder = 6
        self.right_elbow = 7
        self.right_wrist = 8
        self.human_rest_poses = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

        # initial and target human arm
        self.q_human_init = [-1.4, 3.1, 0, 0.1]
        self.q_human_goal = [0, 3.1, 0, 0.1]

        # compute human transforms for constraint func
        human_base = self.bc.getBasePositionAndOrientation(self.humanoid._humanoid)[:2]
        self.T_world_to_human_base = compute_matrix(translation=human_base[0], rotation=human_base[1])
        right_elbow_pose = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[4:6]
        right_elbow_pose_inverse = self.bc.invertTransform(right_elbow_pose[0], right_elbow_pose[1])
        cp_pose = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        right_elbow_to_cp = self.bc.multiplyTransforms(right_elbow_pose_inverse[0], right_elbow_pose_inverse[1],
                                                       cp_pose[0], cp_pose[1])
        self.T_right_elbow_to_cp = compute_matrix(translation=right_elbow_to_cp[0], rotation=right_elbow_to_cp[1])

        # load robot
        self.robot_base_pose = ((0.5, 0.7, 0), (0, 0, 0))
        self.world_to_robot_base = compute_matrix(translation=self.robot_base_pose[0], rotation=self.robot_base_pose[1], rotation_type='euler')
        self.robot = UR5Robotiq85(self.bc, self.robot_base_pose[0], self.robot_base_pose[1])
        self.robot.load()
        self.robot.reset()
        self.robot.open_gripper()

        # initialize robot parameters
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        world_to_eef_grasp = [
                [world_to_eef[0][0], world_to_eef[0][1], world_to_eef[0][2]-0.06],
                world_to_eef[1]
            ]
        eef_grasp_to_world = self.bc.invertTransform(world_to_eef_grasp[0], world_to_eef_grasp[1])
        eef_grasp_to_eef = self.bc.multiplyTransforms(eef_grasp_to_world[0], eef_grasp_to_world[1],
                                                           world_to_eef[0], world_to_eef[1])
        self.eef_grasp_to_eef = eef_grasp_to_eef

        ######### DEBUGGING GRASPS
        # init human arm and get its point cloud
        # # self.reset_human_arm(self.q_human_init)
        # right_elbow_pcd = get_point_cloud_from_collision_shapes_specific_link(self.humanoid._humanoid, self.right_elbow, resolution=20)
        # right_wrist_pcd = get_point_cloud_from_collision_shapes_specific_link(self.humanoid._humanoid, self.right_wrist, resolution=20)
        # pcd = np.vstack((right_elbow_pcd, right_wrist_pcd))
        # dict = {'xyz': pcd}
        # np.save("pc_human_arm.npy", dict)

        # # compute grasps
        # global_config = load_config(checkpoint_dir="contact_graspnet_pytorch/checkpoints/contact_graspnet", batch_size=1, arg_configs=[])
        # inference(global_config=global_config, 
        #           ckpt_dir="contact_graspnet_pytorch/checkpoints/contact_graspnet",
        #           input_paths="pc_human_arm.npy", visualize_results=False)
        grasp_results = np.load("results/predictions_pc_human_arm.npz", allow_pickle=True)
        pred_grasps_cam = grasp_results['pred_grasps_cam'].item()
        scores = grasp_results['scores'].item()
        pred_grasps_cam_values = list(pred_grasps_cam.values())[0]
        scores_values = list(scores.values())[0]

        # compute x_range, y_range, z_range
        right_elbow_world_pos = np.array(self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[0])
        right_elbow_link_pos = np.array(self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[4])
        right_elbow_diff = np.abs(right_elbow_world_pos-right_elbow_link_pos)
        sorted_indices = np.argsort(right_elbow_diff)
        smallest_two_indices = sorted_indices[:2]
        other_index = sorted_indices[-1]

        if 0 in smallest_two_indices:
            x_range = [right_elbow_world_pos[0], right_elbow_world_pos[0]+0.15]
        if 1 in smallest_two_indices:
            y_range = [right_elbow_world_pos[1]-0.15, right_elbow_world_pos[1]+0.15]
        if 2 in smallest_two_indices:
            z_range = [right_elbow_world_pos[2]-0.15, right_elbow_world_pos[2]+0.15]

        if 0 == other_index:
            x_range = [right_elbow_world_pos[0]-0.01, right_elbow_world_pos[0]+right_elbow_diff[0]/6]
        if 1 == other_index:
            y_range = [right_elbow_world_pos[1]-0.01, right_elbow_world_pos[1]+right_elbow_diff[1]/6]
        if 2 == other_index:
            z_range = [right_elbow_world_pos[2]-0.01, right_elbow_world_pos[2]+right_elbow_diff[2]/6]

        # filter matrices based on the ranges
        filtered_matrices, filtered_indices = filter_transform_matrices_by_position(pred_grasps_cam_values, x_range, y_range, z_range)
        top_indices = np.argsort(scores_values[filtered_indices])[-5:][::-1]
        top_transformation_matrices = pred_grasps_cam_values[top_indices]
        print(f'filtered grasps: {len(top_transformation_matrices)}')

        # rotate matrices by 'y' axis
        world_to_right_elbow = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        right_elbow_to_world = self.bc.invertTransform(world_to_right_elbow[0], world_to_right_elbow[1])
        
        grasp_samples = []
        for i in range(len(top_transformation_matrices)):
            # position = top_transformation_matrices[i][:3,3]
            # quaternion = quaternion_from_matrix(top_transformation_matrices[i][:3,:3])

            # # rotate matrices by 'y' axis of human arm
            # right_elbow_to_grasp = self.bc.multiplyTransforms(right_elbow_to_world[0], right_elbow_to_world[1],
            #                                                   position, quaternion)
            # world_to_right_elbow_rotated = [world_to_right_elbow[0], rotate_quaternion_by_axis(world_to_right_elbow[1], axis='y', degrees=180)]
            # world_to_grasp_rotated = self.bc.multiplyTransforms(world_to_right_elbow_rotated[0], world_to_right_elbow_rotated[1],
            #                                                     right_elbow_to_grasp[0], right_elbow_to_grasp[1])

            # # rotate matrices such that panda eef --> ur5 eef
            # position = world_to_grasp_rotated[0]
            # quaternion = world_to_grasp_rotated[1]
            # quaternion = rotate_quaternion_by_axis(quaternion, axis='y', degrees=-90)
            # quaternion_1 = rotate_quaternion_by_axis(quaternion, axis='x', degrees=180)
            # quaternion_2 = rotate_quaternion_by_axis(quaternion, axis='x', degrees=-90)
            # draw_frame(self, position=position, quaternion=quaternion_1)

            # grasp_samples.append([list(position), list(quaternion_1)])
            # grasp_samples.append([list(position), list(quaternion_2)])

            # append more grasps for human arm in this axis
            if 0 == other_index:
                position = top_transformation_matrices[i][:3,3]
                quaternion = quaternion_from_matrix(top_transformation_matrices[i][:3,:3])

                # rotate matrices by 'y' axis of human arm
                right_elbow_to_grasp = self.bc.multiplyTransforms(right_elbow_to_world[0], right_elbow_to_world[1],
                                                                position, quaternion)
                world_to_right_elbow_rotated = [world_to_right_elbow[0], rotate_quaternion_by_axis(world_to_right_elbow[1], axis='y', degrees=90)]
                world_to_grasp_rotated = self.bc.multiplyTransforms(world_to_right_elbow_rotated[0], world_to_right_elbow_rotated[1],
                                                                    right_elbow_to_grasp[0], right_elbow_to_grasp[1])

                # rotate matrices such that panda eef --> ur5 eef
                position = world_to_grasp_rotated[0]
                quaternion = world_to_grasp_rotated[1]
                quaternion = rotate_quaternion_by_axis(quaternion, axis='y', degrees=-90)
                quaternion_1 = rotate_quaternion_by_axis(quaternion, axis='x', degrees=180)
                quaternion_2 = rotate_quaternion_by_axis(quaternion, axis='x', degrees=-90)
                draw_frame(self, position=position, quaternion=quaternion_1)

                grasp_samples.append([list(position), list(quaternion_1)])
                grasp_samples.append([list(position), list(quaternion_2)])

        for grasp in grasp_samples:
            world_to_eef = self.bc.multiplyTransforms(grasp[0], grasp[1],
                                                      self.eef_grasp_to_eef[0], self.eef_grasp_to_eef[1])
            q_R_grasp = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, 
                                                       world_to_eef[0], world_to_eef[1],
                                                    #    grasp[0], grasp[1],
                                                       self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                       maxNumIterations=20)
            q_R_grasp_test = [q_R_grasp[i] for i in range(len(self.robot.arm_controllable_joints))]
            self.reset_robot(self.robot, q_R_grasp_test)
            print('')

        print('here')
        #########

        # load second robot
        self.robot_2_base_pose = ((-0.5, 0.5, 0), (0, 0, 0))
        self.robot_2 = UR5Robotiq85(self.bc, self.robot_2_base_pose[0], self.robot_2_base_pose[1])
        self.robot_2.load()
        self.robot_2.reset()

        # move second robot to sphere obstacle position
        q_robot_2 = self.bc.calculateInverseKinematics(self.robot_2.id, self.robot_2.eef_id, [0.17, 0.2, 0.77],
                                                    self.robot_2.arm_lower_limits, self.robot_2.arm_upper_limits, self.robot_2.arm_joint_ranges, self.robot_2.arm_rest_poses,
                                                    maxNumIterations=20)
        self.q_robot_2 = [q_robot_2[i] for i in range(len(self.robot_2.arm_controllable_joints))]

        for _ in range(10):
            self.reset_robot(self.robot_2, self.q_robot_2)
            self.bc.stepSimulation()

        # # initialize obstacles
        self.obstacles = []
        self.obstacles.append(self.bed_id)
        self.obstacles.append(self.robot_2.id)
        # self.obstacles.append(self.humanoid._humanoid)

        # get 'static' obstacle point cloud
        self.obs_pcd = self.get_obstacle_point_cloud(self.obstacles)

        # initialize robot config
        self.init_robot_configs()
        self.set_robot_target_joint()

        # initialize collision checker
        self.collision_fn = get_collision_fn(self.robot.id, self.robot.arm_controllable_joints, obstacles=self.obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    def init_robot_configs(self):
        # initial human arm and reset robot config
        self.robot.reset()
        self.reset_human_arm(self.q_human_init)
        self.bc.stepSimulation()

        # move robot to grasp pose
        pos = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[0]
        pos_up_2 = (pos[0]+0.1, pos[1]+0.3, pos[2]+0.3)
        pos_up = (pos[0]+0.015, pos[1]+0.06, pos[2])
        orn_2 = self.bc.getQuaternionFromEuler((3.0, 0, -1.57))
        orn = self.bc.getQuaternionFromEuler((3.14, 0, -1.57))

        # pos_up_2
        current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, pos_up_2, orn_2,
                                                    self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                    maxNumIterations=20)
        current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
        for _ in range(50):
            self.reset_robot(self.robot, current_joint_angles)
            self.reset_human_arm(self.q_human_init)
            self.bc.stepSimulation()

        # save joint angles --> q_R_init_before_grasp
        q_R_init_before_grasp = []
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            q_R_init_before_grasp.append(self.bc.getJointState(self.robot.id, joint_id)[0])
        self.q_R_init_before_grasp = q_R_init_before_grasp
        print('q_R_init_before_grasp', self.q_R_init_before_grasp)
        
        # pos_up
        draw_frame(self, pos_up, orn)
        current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, pos_up, orn,
                                                       self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                       maxNumIterations=20)
        current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]

        for _ in range(100):
            self.reset_human_arm(self.q_human_init)
            self.reset_robot(self.robot, current_joint_angles)
            self.bc.stepSimulation()
        
        # save joint angles --> q_R_goal_before_grasp
        q_R_goal_before_grasp = []
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            q_R_goal_before_grasp.append(self.bc.getJointState(self.robot.id, joint_id)[0])
        self.q_R_goal_before_grasp = q_R_goal_before_grasp
        self.q_R_init_after_grasp = q_R_goal_before_grasp
        print('q_R_goal_before_grasp', self.q_R_goal_before_grasp)
        print('q_R_init_after_grasp', self.q_R_init_after_grasp)

        # compute eef_to_cp
        body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)              # world to eef 
        obj_pose = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)      # world to cp
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])             # eef to world
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],                      # eef to cp
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])
        self.eef_to_cp = obj_to_body

    def set_robot_target_joint(self):
        # move human to target config
        self.robot.reset()
        self.reset_human_arm(self.q_human_goal)
        self.bc.stepSimulation()

        world_to_cp = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        cp_to_world = self.bc.invertTransform(world_to_cp[0], world_to_cp[1])
        eef_to_world = self.bc.multiplyTransforms(self.eef_to_cp[0], self.eef_to_cp[1],
                                                  cp_to_world[0], cp_to_world[1])
        world_to_eef = self.bc.invertTransform(eef_to_world[0], eef_to_world[1])
        draw_frame(self, position=world_to_eef[0], quaternion=world_to_eef[1])
        self.target_eef = world_to_eef
        
        current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                       self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                       maxNumIterations=20)
        current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]

        # save joint angles --> q_R_goal_after_grasp
        self.q_R_goal_after_grasp = current_joint_angles
        
    def update_pcd(self):
        link_to_separate = [self.right_elbow, self.right_wrist]
        human_pcd, separate_pcd = get_humanoid_point_cloud(self.humanoid._humanoid, link_to_separate)
        self.obs_pcd = np.vstack((self.obs_pcd, human_pcd))
        self.right_arm_pcd = np.array(separate_pcd)

        # self.visualize_point_cloud(self.obs_pcd)
        # self.visualize_point_cloud(self.right_arm_pcd)
    
        # update environment point cloud
        self.trajectory_planner.update_obstacle_pcd(self.obs_pcd)
        print("Updated environment point cloud")

    ### INITIALIZE PLANNER
    def init_traj_planner(self):
        JOINT_LIMITS = [
            np.array(self.robot.arm_lower_limits), 
            np.array(self.robot.arm_upper_limits)
        ]

        # order for humanoid urdf: [yaw, pitch, roll]
        # shoulder_min = [-3.1402077384521765, -0.248997453133789, -2.583238756496965]
        # shoulder_max = [3.1415394736319917, 1.2392816988875348, -1.3229245882839409]
        # shoulder_min = [-2.583238756496965, -0.248997453133789, -3.1402077384521765]
        # shoulder_max = [-1.3229245882839409, 1.2392816988875348, 3.1415394736319917]
        shoulder_min = [-3.14, -3.14, -3.14]
        shoulder_max = [3.14, 3.14, 3.14]
        elbow_min = [0.401146]
        # elbow_min = [0]
        elbow_max = [2.541304]  
        self.human_arm_lower_limits = shoulder_min + elbow_min
        self.human_arm_upper_limits = shoulder_max + elbow_max

        # Instantiate mppi H clamp
        self.mppi_H_clamp = MPPI_H_Clamp(self.eef_to_cp, self.robot_base_pose,
                                         self.human_arm_lower_limits, self.human_arm_upper_limits, human_rest_poses=self.q_human_goal)

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
            mppi_H_clamp = self.mppi_H_clamp,
            world_to_robot_base = self.world_to_robot_base,
        )
        print("Instantiated trajectory planner")

    def init_mppi_planner(self, current_joint_angles, target_joint_angles, clamp_by_human):
        # MPPI parameters
        N_JOINTS = len(self.robot.arm_controllable_joints)
        # mppi_control_limits = [
        #     -0.05 * np.ones(N_JOINTS),
        #     0.05 * np.ones(N_JOINTS)
        # ]
        # mppi_nsamples = 1000
        # mppi_covariance = 0.005
        # mppi_lambda = 1.0
        mppi_control_limits = [
            -0.05 * np.ones(N_JOINTS),
            0.05 * np.ones(N_JOINTS)
        ]
        mppi_nsamples = 500
        mppi_covariance = 0.005
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
        self.trajectory_planner.update_obstacle_pcd(self.obs_pcd)
        print("Updated environment point cloud")

    def get_mppi_trajectory(self, current_joint_angles):
        # Plan trajectory
        start_time = time.time()
        trajectory = self.trajectory_planner.get_mppi_rollout(current_joint_angles)
        print("planning time : ", time.time()-start_time)
        print(np.array(trajectory))
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
            world_to_robot_base = self.world_to_robot_base,
        )
        print('trajectory follower instantiated')

        # TODO update environment point cloud
        self.trajectory_follower.update_obstacle_pcd(self.obs_pcd)
        print("Updated environment point cloud")

    def init_informed_rrtstar_planner(self, current_joint_angles, target_joint_angles, traj):
        self.informed_rrtstar_planner = InformedRRTStar(current_joint_angles, target_joint_angles, self.obstacles,
                                                        self.robot.id, self.robot.arm_controllable_joints)
        return self.informed_rrtstar_planner.plan(traj)

    ### HELPER FUNCTIONS
    def attach_human_arm_to_eef(self, joint_type=p.JOINT_FIXED, attach_to_gripper=False):
        # attach human arm (obj) to eef (body)
        body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]            # world to eef 
        obj_pose = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]    # world to cp
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])               # eef to world
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],                        # eef to cp
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])
        self.eef_to_cp = obj_to_body
        
        self.cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                            parentLinkIndex=self.robot.eef_id,
                            childBodyUniqueId=self.humanoid._humanoid,
                            childLinkIndex=self.right_elbow,
                            jointType=joint_type,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=self.eef_to_cp[0],
                            parentFrameOrientation=self.eef_to_cp[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))

        if attach_to_gripper:
            # compute transform matrix from robot's gripper to object frame
            T_eef_to_object = compute_matrix(translation=self.eef_to_cp[0], rotation=self.eef_to_cp[1], rotation_type='quaternion')

            # compute transform matrix for inverse of object pose in world frame
            T_world_to_object = compute_matrix(translation=obj_pose[0], rotation=obj_pose[1], rotation_type='quaternion')
            T_object_to_world = inverse_matrix(T_world_to_object)

            self.trajectory_planner.attach_to_gripper(object_type="pcd", object_geometry=self.right_arm_pcd,
                                                      T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world)

            return T_eef_to_object, T_object_to_world
    
    def deattach_human_arm_from_eef(self):
        if self.cid is not None:
            self.bc.removeConstraint(self.cid)
        self.cid = None

    def visualize_point_cloud(self, pcd):
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pcd)
        o3d.visualization.draw_geometries([pc_ply])

    def draw_sphere_marker(self, position, radius=0.07, color=[1, 0, 0, 1]):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        col_id = self.bc.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vs_id)
        return marker_id 

    def human_motion_from_frame_data(self, humanoid, motion, utNum):
        keyFrameDuration = motion.KeyFrameDuraction()
        self.bc.stepSimulation()
        humanoid.RenderReference(utNum * keyFrameDuration, self.bc)
        self.bc.stepSimulation()

    def reset_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.resetJointState(robot.id, joint_id, q_robot[i])

    def move_robot(self, robot, q_robot):
        for i, joint_id in enumerate(robot.arm_controllable_joints):
            self.bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i])

    def reset_human_arm(self, q_human):
        self.bc.resetJointState(self.humanoid._humanoid, self.right_shoulder_y, q_human[0])
        self.bc.resetJointState(self.humanoid._humanoid, self.right_shoulder_p, q_human[1])
        self.bc.resetJointState(self.humanoid._humanoid, self.right_shoulder_r, q_human[2])
        self.bc.resetJointState(self.humanoid._humanoid, self.right_elbow, q_human[3])

    def move_human_arm(self, q_human):
        self.bc.setJointMotorControl2(self.humanoid._humanoid, self.right_shoulder_y, p.POSITION_CONTROL, q_human[0])
        self.bc.setJointMotorControl2(self.humanoid._humanoid, self.right_shoulder_p, p.POSITION_CONTROL, q_human[1])
        self.bc.setJointMotorControl2(self.humanoid._humanoid, self.right_shoulder_r, p.POSITION_CONTROL, q_human[2])
        self.bc.setJointMotorControl2(self.humanoid._humanoid, self.right_elbow, p.POSITION_CONTROL, q_human[3])

    def get_obstacle_point_cloud(self, obstacles):
        point_cloud = []
        for obstacle in obstacles:
            if obstacle == self.humanoid._humanoid:
                human_pcd, _ = get_humanoid_point_cloud(self.humanoid._humanoid, [self.right_elbow])
                point_cloud.extend(human_pcd)
            elif obstacle == self.bed_id:
                half_extents = [0.5, 1.7, 0.2]
                point_cloud.extend(get_point_cloud_from_collision_shapes(obstacle, half_extents))
            else:
                point_cloud.extend(get_point_cloud_from_visual_shapes(obstacle))
        return np.array(point_cloud)
    
    def update_current_joint_angles(self, current_joint_angles):
        self.current_joint_angles = current_joint_angles

    def update_target_joint_angles(self, target_joint_angles):
        self.target_joint_angles = target_joint_angles

    def is_near_goal(self, current_joint_angles):
        dist = np.linalg.norm(np.array(self.q_R_goal_after_grasp) - np.array(current_joint_angles))
        print(dist)
        if dist <= 0.75:
            return True
        else:
            return False

if __name__ == '__main__':
    env = HumanDemo()
    env.init_traj_planner()
    env.init_traj_follower()

    # ####

    # # Step 0: trajectory before grasping
    # traj = env.init_mppi_planner(env.q_R_init_before_grasp, env.q_R_goal_before_grasp, clamp_by_human=False)
    # print(traj)

    # for q in traj:
    #     for _ in range(100):
    #         env.move_robot(env.robot, q)
    #         env.move_human_arm(env.q_human_init)
    #         env.bc.stepSimulation()
    #     time.sleep(0.5)

    # for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #     print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

    #####

    # TEST MOVING WITH HUMAN ARM ATTACHED

    # Step 1: move robot to grasping pose
    for _ in range(100):
        env.reset_robot(env.robot_2, env.q_robot_2)
        env.reset_robot(env.robot, env.q_R_goal_before_grasp)
        env.reset_human_arm(env.q_human_init)
        env.bc.stepSimulation()

    # Step 2: attach human arm to eef
    env.update_pcd()
    T_eef_to_object, T_object_to_world = env.attach_human_arm_to_eef(attach_to_gripper=True)

    # Step 3: trajectory after grasping
    env.init_mppi_planner(env.q_R_goal_before_grasp, env.q_R_goal_after_grasp, clamp_by_human=True)
    traj = env.get_mppi_trajectory(env.q_R_goal_before_grasp)
    previous_update_time = time.time()
    update_second = 10  # sec

    # Step 4: initialize trajectory follower
    # right_shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_shoulder)
    # pcd = np.vstack((env.obs_pcd, right_shoulder_pcd))
    # env.trajectory_follower.update_obstacle_pcd(pcd)
    # env.visualize_point_cloud(env.obs_pcd)
    env.trajectory_follower.update_obstacle_pcd(env.obs_pcd)
    env.trajectory_follower.update_trajectory(traj)
    env.trajectory_follower.attach_to_gripper(object_type="pcd", object_geometry=env.right_arm_pcd,
                                              T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world,
                                              T_world_to_human_base=env.T_world_to_human_base, T_right_elbow_to_cp=env.T_right_elbow_to_cp)
    current_joint_angles = env.q_R_goal_before_grasp
    current_human_joint_angles = env.q_human_init

    # env.deattach_human_arm_from_eef()
    # env.reset_human_arm(env.q_human_goal)

    # Step 5: simulation loop
    while True:
        # if near goal, execute rest of trajectory and end simulation loop
        if env.is_near_goal(current_joint_angles):
            for q in traj:
                for _ in range (100):
                    env.reset_robot(env.robot_2, env.q_robot_2)
                    env.move_robot(env.robot, q)
                    env.bc.stepSimulation()
                time.sleep(0.5)
            break

        # get velocity command
        prev_time = time.time()
        velocity_command = env.trajectory_follower.follow_trajectory(current_joint_angles, current_human_joint_angles)[0]
        current_time = time.time()
        print('following time: ', current_time-prev_time)

        # update trajectory 
        if current_time-previous_update_time > update_second:
            print('replanning...')
            # right_shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_shoulder)
            # pcd = np.vstack((env.obs_pcd, right_shoulder_pcd))
            # env.trajectory_planner.update_obstacle_pcd(pcd)
            # env.trajectory_follower.update_obstacle_pcd(pcd)

            traj = env.get_mppi_trajectory(current_joint_angles)
            env.trajectory_follower.update_trajectory(traj)
            previous_update_time = time.time()

        # if valid velocity command, move robot
        else:
            for _ in range(100):
                for i, joint_id in enumerate(env.robot.arm_controllable_joints):
                    env.bc.setJointMotorControl2(
                            env.robot.id, joint_id,
                            controlMode = p.VELOCITY_CONTROL,
                            targetVelocity = velocity_command[i]
                        )
                env.reset_robot(env.robot_2, env.q_robot_2)
                env.bc.stepSimulation()

            # save current_joint_angle
            current_joint_angles = []
            for joint_id in env.robot.arm_controllable_joints:
                current_joint_angles.append(env.bc.getJointState(env.robot.id, joint_id)[0])
            current_human_joint_angles = []
            current_human_joint_angles.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_y)[0])
            current_human_joint_angles.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_p)[0])
            current_human_joint_angles.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_r)[0])
            current_human_joint_angles.append(env.bc.getJointState(env.humanoid._humanoid, env.right_elbow)[0])

    time.sleep(10)
    print('done')
    ###

    # ## TEST MOVING WITHOUT HUMAN ARM ATTACHED

    # # Step 1: move robot to grasping pose
    # for _ in range(100):
    #     env.reset_robot(env.robot_2, env.q_robot_2)
    #     env.reset_robot(env.robot, env.q_R_goal_before_grasp)
    #     env.bc.stepSimulation()

    # # Step 2: initialize trajectory planner and get mppi trajectory (after grasping)
    # env.init_mppi_planner(env.q_R_goal_before_grasp, env.q_R_goal_after_grasp, clamp_by_human=False)
    # traj = env.get_mppi_trajectory(env.q_R_goal_before_grasp)
    # previous_update_time = time.time()
    # update_second = 5  # sec

    # # Step 3: initialize trajectory follower
    # env.init_traj_follower()
    # env.trajectory_follower.update_trajectory(traj)
    # current_joint_angles = env.q_R_goal_before_grasp

    # # Step 4: simulation loop
    # while True:
    #     # get velocity command
    #     prev_time = time.time()
    #     velocity_command = env.trajectory_follower.follow_trajectory(current_joint_angles)[0]
    #     current_time = time.time()
    #     print('following time: ', current_time-prev_time)

    #     # update trajectory 
    #     if current_time-previous_update_time > update_second:
    #         traj = env.get_mppi_trajectory(current_joint_angles)
    #         env.trajectory_follower.update_trajectory(traj)
    #         previous_update_time = time.time()

    #     # if valid velocity command, move robot
    #     else:
    #         for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #             env.bc.setJointMotorControl2(
    #                     env.robot.id, joint_id,
    #                     controlMode = p.VELOCITY_CONTROL,
    #                     targetVelocity = velocity_command[i]
    #                 )
    #         env.reset_robot(env.robot_2, env.q_robot_2)
    #         env.bc.stepSimulation()

    #         # save current_joint_angle
    #         current_joint_angles = []
    #         for joint_id in env.robot.arm_controllable_joints:
    #             current_joint_angles.append(env.bc.getJointState(env.robot.id, joint_id)[0])

    # #####

    ## DEBUGGING TRAJ
    
    traj = np.array([[-2.25131376, -2.75403316,  2.47657261, -1.63770982, -1.50791089,  0.57950216],
 [-2.2594125 , -2.69183667,  2.3577178 , -1.52532379, -1.42537428,  0.60821025],
 [-2.22364404, -2.59686807,  2.23862245, -1.4458647 , -1.37892938,  0.6894849 ],
 [-2.17368458, -2.49006036,  2.13829055, -1.40708691, -1.35863832,  0.801433  ],
 [-2.11583582, -2.39251118,  2.05244401, -1.41475413, -1.32953978,  0.9377891 ],
 [-2.07636324, -2.2846245 ,  1.9755482 , -1.43004539, -1.32985656,  1.07260082],
 [-2.05432664, -2.17579465,  1.86233269, -1.39655633, -1.28538235,  1.17735241],
 [-2.03296581, -2.06952079,  1.77678745, -1.42388806, -1.26662067,  1.31219035],
 [-2.01610716, -1.96339687,  1.69127296, -1.45882471, -1.2582444 ,  1.45040615],
 [-1.9936056 , -1.86636543,  1.63034108, -1.52659069, -1.27485072,  1.59495001],
 [-1.96879363, -1.74494649,  1.55633625, -1.53556323, -1.32527522,  1.72132378],
 [-1.97162471, -1.62394717,  1.45554699, -1.52843886, -1.3488897 ,  1.8383477 ],
 [-2.0077383 , -1.5123799 ,  1.3388635 , -1.51697911, -1.34439535,  1.94882628],
 [-2.04228275, -1.41022548,  1.2489475 , -1.53729084, -1.37900382,  2.07194629],
 [-2.07510526, -1.32428931,  1.15083346, -1.58101515, -1.4131321 ,  2.19193319],
 [-2.15737589, -1.22944521,  1.09776156, -1.59380365, -1.42646212,  2.32692557],
 [-2.23964653, -1.13460112,  1.04468966, -1.60659215, -1.43979213,  2.46191796],
 [-2.24597504, -1.12730542,  1.04060721, -1.60757588, -1.44081752,  2.47230199]]
)

    # for _ in range (300):
    #     env.reset_robot(env.robot_2, env.q_robot_2)
    #     env.move_robot(env.robot, traj[0])
    #     env.bc.stepSimulation()
    # env.attach_human_arm_to_eef()

    # time.sleep(4)

    # # move to this trajectory
    # for q in traj:
    #     for _ in range (300):
    #         env.reset_robot(env.robot_2, env.q_robot_2)
    #         env.move_robot(env.robot, q)
    #         env.bc.stepSimulation()
    #     # check for collision
    #     if env.collision_fn(q):
    #         print('collision!')
    #     time.sleep(0.5)


    # initialize trajectory follower
    env.init_traj_follower()
    env.trajectory_follower.update_trajectory(traj)
    current_joint_angles = traj[0]

    # move robot to grasping pose
    for _ in range(100):
        env.reset_robot(env.robot_2, env.q_robot_2)
        env.reset_robot(env.robot, env.q_R_goal_before_grasp)
        env.reset_human_arm(env.q_human_init)
        env.bc.stepSimulation()

    # update point cloud (attach arm control points)
    env.update_pcd()
    T_eef_to_object, T_object_to_world = env.attach_human_arm_to_eef(attach_to_gripper=True)
    right_shoulder_pcd = get_point_cloud_from_collision_shapes_specific_link(env.humanoid._humanoid, env.right_shoulder)
    pcd = np.vstack((env.obs_pcd, right_shoulder_pcd))
    env.trajectory_follower.update_obstacle_pcd(pcd)
    env.trajectory_follower.attach_to_gripper(object_type="pcd", object_geometry=env.right_arm_pcd,
                                              T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world,
                                              T_world_to_human_base=env.T_world_to_human_base, T_right_elbow_to_cp=env.T_right_elbow_to_cp)
    # env.deattach_human_arm_from_eef()

    # move to start following the traj
    for _ in range (500):
        env.reset_robot(env.robot_2, env.q_robot_2)
        env.move_robot(env.robot, traj[0])
        env.bc.stepSimulation()
    print('here')

    while True:
        # get velocity command
        velocity_command = env.trajectory_follower.follow_trajectory(current_joint_angles, [])[0]
        current_time = time.time()

        # if valid velocity command, move robot
        for i, joint_id in enumerate(env.robot.arm_controllable_joints):
            env.bc.setJointMotorControl2(
                    env.robot.id, joint_id,
                    controlMode = p.VELOCITY_CONTROL,
                    targetVelocity = velocity_command[i]
                )
        env.reset_robot(env.robot_2, env.q_robot_2)
        env.bc.stepSimulation()

        # save current_joint_angle
        current_joint_angles = []
        for joint_id in env.robot.arm_controllable_joints:
            current_joint_angles.append(env.bc.getJointState(env.robot.id, joint_id)[0])