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
# from contact_graspnet_pytorch.contact_graspnet_pytorch.inference import *
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
        self.bc.setTimestep = 0.0005

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, -0.04, 0))
        bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0, 0.0), useFixedBase=True)  # bed
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
        self.right_elbow = 7
        self.human_rest_poses = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

        # initial and target human arm
        self.q_human_init = [-1.4, 3.1, 0, 0.1]
        self.q_human_goal = [0, 3.1, 0, 0.1]

        # # move human to initial config
        # self.reset_human_arm(self.q_human_goal)
        # self.bc.stepSimulation()

        # load robot
        self.robot_base_pose = ((0.5, 0.7, 0), (0, 0, 0))
        self.world_to_robot_base = compute_matrix(translation=self.robot_base_pose[0], rotation=self.robot_base_pose[1], rotation_type='euler')
        self.robot = UR5Robotiq85(self.bc, self.robot_base_pose[0], self.robot_base_pose[1])
        self.robot.load()
        self.robot.reset()
        self.robot.open_gripper()

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

        # # add sphere obstacle
        # sphere_pos1 = (0.15, 0.5, 0.5)
        # sphere_pos2 = (0.17, 0.3, 0.8)
        # self.sphere1 = self.draw_sphere_marker(sphere_pos1, radius=0.05)
        # self.sphere2 = self.draw_sphere_marker(sphere_pos2, radius=0.05)

        # # get point cloud
        # point_cloud = []
        # # point_cloud.extend(get_point_cloud_from_visual_shapes(self.robot.id))
        # point_cloud.extend(get_point_cloud_from_visual_shapes(self.sphere1))
        # point_cloud.extend(get_point_cloud_from_visual_shapes(self.sphere2))
        # # point_cloud.extend(get_point_cloud_from_visual_shapes(bed_id))
        # self.obs_pcd = np.array(point_cloud)

        # ### DEBUGGING
        # draw_control_points(self)

        # # initialize obstacles
        self.obstacles = []
        self.obstacles.append(bed_id)
        self.obstacles.append(self.robot_2.id)
        self.obstacles.append(self.humanoid._humanoid)
        # self.obstacles.append(self.sphere2)

        # get obstacle point cloud
        self.obs_pcd = self.get_obstacle_point_cloud(self.obstacles)
        self.visualize_point_cloud(self.obs_pcd)

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
        # draw_frame(self, position=world_to_eef[0], quaternion=world_to_eef[1])
        
        current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                       self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                       maxNumIterations=20)
        current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]

        # save joint angles --> q_R_goal_after_grasp
        self.q_R_goal_after_grasp = current_joint_angles
        
    def update_pcd(self):
        human_pcd, right_elbow_pcd = get_humanoid_point_cloud(self.humanoid._humanoid, self.right_elbow)
        self.obs_pcd = np.vstack((self.obs_pcd, human_pcd))
        self.right_elbow_pcd = np.array(right_elbow_pcd)
    
        # # update environment point cloud
        # self.trajectory_planner.update_obstacle_pcd(self.obs_pcd)
        # print("Updated environment point cloud")

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
        # elbow_min = [0.401146]
        elbow_min = [0]
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
        mppi_control_limits = [
            -0.5 * np.ones(N_JOINTS),
            0.5 * np.ones(N_JOINTS)
        ]
        mppi_nsamples = 500
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
        self.trajectory_planner.update_obstacle_pcd(self.obs_pcd)
        print("Updated environment point cloud")

        # Plan trajectory
        start_time = time.time()
        trajectory = self.trajectory_planner.get_mppi_rollout(self.current_joint_angles)
        print("planning time : ", time.time()-start_time)
        return trajectory
    
    def init_informed_rrtstar_planner(self, current_joint_angles, target_joint_angles, traj):
        self.informed_rrtstar_planner = InformedRRTStar(current_joint_angles, target_joint_angles, self.obstacles,
                                                        self.robot.id, self.robot.arm_controllable_joints)
        return self.informed_rrtstar_planner.plan(traj)
    
    ### CAMERA IMAGE
    def get_camera_img(self):
        ### Desired image size and intrinsic camera parameters
        ### Intrinsic camera matrix from 4th Oct 2021:    543.820     0.283204    314.424;
        ###                                               0.00000     546.691     237.466;
        width = 480
        height = 480
        f_x = 543.820
        f_y = 546.691
        c_x = 314.424
        c_y = 237.466
        skew = 0.283204

        ### The far and near values depend on the min,max desired distance from the object
        far = 1.5
        near = 0.1

        ### Camera intrinsic:
        opengl_projection_matrix = (f_x/width,            0,                      0,                      0,
                                    skew/width,           f_y/height,           0,                      0,
                                    2*(c_x+0.5)/width-1,  2*(c_y+0.5)/height-1,   -(far+near)/(far-near), -1,
                                    0,                    0,                      -2*far*near/(far-near), 0)

        obj_pb_id = self.humanoid._humanoid
        obj_link_id = self.right_elbow

        aabb = p.getAABB(obj_pb_id, obj_link_id, physicsClientId=self.bc._client)
        aabb_min = np.array(aabb[0])
        aabb_max = np.array(aabb[1])
        obj_center = list((aabb_max + aabb_min)/2)

        camera_look_at = obj_center

        phi_deg = 130
        theta_deg = 15
        camera_distance = 0.6

        phi = np.deg2rad(phi_deg)
        theta = np.deg2rad(theta_deg)
        camera_eye_position = []
        camera_eye_position.append(camera_distance*np.cos(phi)*np.sin(theta) + obj_center[0])
        camera_eye_position.append(camera_distance*np.sin(phi)*np.sin(theta) + obj_center[1])
        camera_eye_position.append(camera_distance*np.cos(theta) + obj_center[2])

        # camera_eye_position.append(obj_center[0])
        # camera_eye_position.append(obj_center[1])
        # camera_eye_position.append(1.0 + obj_center[2])

        view_matrix = self.bc.computeViewMatrix(
            cameraEyePosition= camera_eye_position,
            cameraTargetPosition=camera_look_at,
            cameraUpVector = [0,0,1],
            physicsClientId=self.bc._client
        )

        # T_world_to_camera = np.array(view_matrix).reshape(4,4)
        # T_world_to_camera = T_world_to_camera.T
        # world_to_camera_pos = -T_world_to_camera[:3, 3]
        # world_to_camera_orn = quaternion_from_matrix(T_world_to_camera)
        # draw_frame(self, position=world_to_camera_pos, quaternion=world_to_camera_orn)
        # T_world_to_camera = compute_matrix(translation=world_to_camera_pos, rotation=world_to_camera_orn, rotation_type='quaternion')

        view_mtx = np.array(view_matrix).reshape((4,4),order='F')
        cam_pos = np.dot(view_mtx[:3,:3].T, -view_mtx[:3,3])


        ### Get depth values using Tiny renderer
        images = self.bc.getCameraImage(height=height,
                width=width,
                viewMatrix=view_matrix,
                projectionMatrix=opengl_projection_matrix,
                shadow=True,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=self.bc._client
                )
        
        rgb = images[2][:, :, :-1]
        seg = np.reshape(images[4], [height,width])
        depth_buffer = np.reshape(images[3], [height,width]) 

        # Define the camera intrinsic matrix
        K = np.array([
            [f_x, skew, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])

        dict = {'rgb': rgb, 'depth': depth_buffer, 'K': K, 'seg': seg}
        np.save("img_human_scene.npy", dict)

        return T_world_to_camera

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

            self.trajectory_planner.attach_to_gripper(object_type="pcd", object_geometry=self.right_elbow_pcd,
                                                      T_eef_to_obj=T_eef_to_object, T_obj_to_world=T_object_to_world)
    
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
            # point_cloud.extend(get_point_cloud_from_collision_shapes(obstacle))
            point_cloud.extend(get_point_cloud_from_visual_shapes(obstacle))
        return np.array(point_cloud)

if __name__ == '__main__':
    env = HumanDemo()
    env.init_traj_planner()

    # # Step 1: trajectory before grasping
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

    # Step 1: move robot to grasping pose
    for _ in range(100):
        env.reset_robot(env.robot_2, env.q_robot_2)
        env.reset_robot(env.robot, env.q_R_goal_before_grasp)
        # env.reset_human_arm(env.q_human_init)
        env.bc.stepSimulation()

    # # # Step 2: attach human arm to eef
    # env.update_pcd()
    # env.attach_human_arm_to_eef(attach_to_gripper=True)

    # Step 3: trajectory after grasping
    traj = env.init_mppi_planner(env.q_R_init_after_grasp, env.q_R_goal_after_grasp, clamp_by_human=False)
    print(traj)

    # # # Step 4: detach human arm
    # env.reset_human_arm(env.q_human_goal)
    # env.bc.stepSimulation()
    # # env.deattach_human_arm_from_eef()
    # # env.bc.stepSimulation()
    # time.sleep(2)


    for q in traj:
        for _ in range (300):
            env.reset_robot(env.robot_2, env.q_robot_2)
            env.move_robot(env.robot, q)
            env.bc.stepSimulation()
        # check for collision
        if env.collision_fn(q):
            print('collision!')
        time.sleep(0.5)

    for i, joint_id in enumerate(env.robot.arm_controllable_joints):
        print(i, env.bc.getJointState(env.robot.id, joint_id)[0])


    # traj = [[-3.11386739e+00, -1.76164691e+00,  1.22416424e+00, -9.02148110e-01,
    #         4.29139712e-04,  1.44108175e+00],
    #         [-3.02930256e+00, -1.85107927e+00 , 1.25371818e+00 ,-9.13523133e-01,
    #         -7.61459340e-02,  1.56562846e+00],
    #         [-2.97705818e+00, -1.87156734e+00  ,1.19430375e+00 ,-9.55770389e-01,
    #         -2.17730051e-01,  1.66687387e+00],
    #         [-2.97786049e+00, -1.82880447e+00  ,1.09617021e+00 ,-1.01394482e+00,
    #         -3.66203472e-01,  1.71600753e+00],
    #         [-2.98104626e+00, -1.84668776e+00  ,1.11795089e+00 ,-1.08453633e+00,
    #         -5.13580688e-01,  1.81942949e+00],
    #         [-2.96571295e+00, -1.84644592e+00  ,1.09028789e+00 ,-1.16176760e+00,
    #         -6.66586610e-01,  1.91575805e+00],
    #         [-2.92668811e+00, -1.82006348e+00  ,9.96716407e-01 ,-1.23314637e+00,
    #         -7.97124254e-01,  1.98257166e+00],
    #         [-2.86032357e+00, -1.82221192e+00  ,8.86122076e-01 ,-1.28758753e+00,
    #         -9.21289069e-01,  2.05231226e+00],
    #         [-2.75466568e+00, -1.82010616e+00  ,8.07137025e-01 ,-1.27900833e+00,
    #         -1.02543020e+00,  2.15902463e+00],
    #         [-2.69048092e+00, -1.74320623e+00  ,7.83813966e-01 ,-1.27842710e+00,
    #         -1.14108037e+00,  2.28495985e+00],
    #         [-2.65527328e+00, -1.63406321e+00  ,8.59301162e-01 ,-1.41809050e+00,
    #         -1.16436828e+00,  2.30464640e+00],
    #         [-2.56695437e+00, -1.54918346e+00  ,9.01788390e-01 ,-1.56339883e+00,
    #         -1.17837075e+00,  2.33550639e+00],
    #         [-2.42963411e+00, -1.51268728e+00  ,8.75448948e-01 ,-1.65558025e+00,
    #         -1.26835904e+00,  2.36422273e+00],
    #         [-2.32878629e+00, -1.49335571e+00  ,8.39883020e-01 ,-1.70037665e+00,
    #         -1.42715402e+00,  2.35313858e+00],
    #         [-2.26123644e+00, -1.37850816e+00  ,8.40335174e-01 ,-1.81531153e+00,
    #         -1.35787235e+00,  2.28914850e+00],
    #         [-2.25349430e+00, -1.26347144e+00  ,9.30423135e-01 ,-1.72420691e+00,
    #         -1.39288734e+00,  2.37010116e+00],
    #         [-2.24747889e+00, -1.15453862e+00  ,1.01857040e+00 ,-1.63090209e+00,
    #         -1.43123148e+00,  2.45186182e+00],
    #         [-2.24597504e+00, -1.12730542e+00  ,1.04060721e+00 ,-1.60757588e+00,
    #         -1.44081752e+00,  2.47230199e+00]]


    # for q in traj:
    #     for _ in range (300):
    #         env.reset_robot(env.robot_2, env.q_robot_2)
    #         env.move_robot(env.robot, q)
    #         env.bc.stepSimulation()
    #     # check for collision
    #     if env.collision_fn(q):
    #         print('collision!')
    #         print(q)
    #     time.sleep(0.5)

    # Step 5: trajectory by informed rrtstar
    for _ in range(100):
        env.reset_robot(env.robot_2, env.q_robot_2)
        env.reset_robot(env.robot, env.q_R_init_after_grasp)
        # env.reset_human_arm(env.q_human_goal)
        env.bc.stepSimulation()

    traj2 = env.init_informed_rrtstar_planner(env.q_R_init_after_grasp, env.q_R_goal_after_grasp, traj)
    print('informed rrtstar planner done')
    print(traj2)

    for q in traj2:
        for _ in range (300):
            env.reset_robot(env.robot_2, env.q_robot_2)
            env.move_robot(env.robot, q)
            env.bc.stepSimulation()
            # check for collision
        if env.collision_fn(q):
            print('collision!')
            time.sleep(3)
        time.sleep(0.5)

    for i, joint_id in enumerate(env.robot.arm_controllable_joints):
        print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

