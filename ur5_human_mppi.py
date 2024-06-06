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

# # contact graspnet
# from contact_graspnet_pytorch.contact_graspnet_pytorch.inference import *

class HumanDemo():
    def __init__(self):
        self.bc = BulletClient(connection_mode=p.GUI)
        # self.bc = BulletClient(connection_mode=p.DIRECT)
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_Y_AXIS_UP, 1)
        self.bc.setGravity(0, -9.8, 0) 
        self.bc.setTimestep = 0.0005

        y2zOrn = self.bc.getQuaternionFromEuler((-1.57, 0, 0))

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)
        bed_id = self.bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed

        # load human
        motionPath = 'data/Sitting1.json'
        self.motion = MotionCaptureData()
        self.motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc, self.motion, [0, 0.3, 0])
        self.right_shoulder_y = 3
        self.right_shoulder_p = 4
        self.right_shoulder_r = 5
        self.right_elbow = 7
        self.human_rest_poses = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]

        # self.bc.changeVisualShape(self.humanoid._humanoid, -1, rgbaColor=[0, 0, 0, 0])
        # self.bc.changeVisualShape(bed_id, -1, rgbaColor=[0, 0, 0, 0])

        ################# DEBUGGING ###################

        pos, orn = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]

        # add sphere
        sphere_pos1 = (pos[0], pos[1]+0.2, pos[2]-0.1)
        self.draw_sphere_marker(sphere_pos1, radius=0.1)

        self.pcd = self.get_point_cloud()

        # load robot
        self.robot = UR5Robotiq85(self.bc, (-0.75, 0, 0), (-1.57, 0, 0), globalScaling=1.2)  # original one
        self.robot.load()
        self.robot.reset()

        left = (pos[0], pos[1]+0.2, pos[2]-0.4)
        right = (pos[0], pos[1]+0.2, pos[2]+0.3)
        self.draw_sphere_marker(left, radius=0.05, color=[0,1,0,1])
        self.draw_sphere_marker(right, radius=0.05, color=[0,0,1,1])

        current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, left,
                                                    self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                    maxNumIterations=20)
        self.current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
        target_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, right,
                                                    self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                    maxNumIterations=20)
        self.target_joint_angles = [target_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
        
        self.move_robot(self.current_joint_angles)

        ################# DEBUGGING ###################

        ##################
        # for joint_id in range(self.bc.getNumJoints(self.humanoid._humanoid)):
        #     print(self.bc.getJointInfo(self.humanoid._humanoid, joint_id))
        ##################

        # # TODO get camera image
        # self.human_motion_from_frame_data(self.humanoid, self.motion, 230)
        # self.get_camera_img()

        # # load robot
        # self.robot = UR5Robotiq85(self.bc, (-0.75, 0, 0), (-1.57, 0, 0), globalScaling=1.2)
        # self.robot.load()
        # self.robot.reset()

        # # TODO load second robot
        # self.robot_2 = UR5Robotiq85(self.bc, (-0.75, 0, 1.0), (-1.57, 0, 0), globalScaling=1.2)
        # self.robot_2.load()
        # self.robot_2.reset()

        # # TODO call inference.py to get grasps

        # # initialize robot config
        # self.init_robot_configs()
        # self.set_robot_target_joint()

    def init_robot_configs(self):
        # initial human arm and reset robot config
        self.human_motion_from_frame_data(self.humanoid, self.motion, 230)
        self.robot.reset()

        # move robot to grasp pose
        pos_up_2 = (-0.17962980178173082, 0.7, 0.1)
        pos_up = (-0.17962980178173082, 0.590458026205689, 0.3667859122611105)
        orn = (0.42581023056381473, 0.025895246484703916, 0.8784134154854197, -0.21541809406808593)

        # pos_up_2
        current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, pos_up_2, orn,
                                                    self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                    maxNumIterations=20)

        for _ in range (100):
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, current_joint_angles[i],
                                                force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
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
        self.current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]

        for _ in range (100):
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, current_joint_angles[i],
                                                force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
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
        self.attach_human_arm_to_eef()

        # move human to target config  
        self.human_motion_from_frame_data(self.humanoid, self.motion, 260)

        # save joint angles --> q_R_goal_after_grasp
        q_R_goal_after_grasp = []
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            q_R_goal_after_grasp.append(self.bc.getJointState(self.robot.id, joint_id)[0])
        self.q_R_goal_after_grasp = q_R_goal_after_grasp
        print('q_R_goal_after_grasp', self.q_R_goal_after_grasp)

        self.deattach_human_arm_from_eef()  

        # move human back to init config
        self.human_motion_from_frame_data(self.humanoid, self.motion, 230)

        # move robot to back to init config
        self.move_robot(self.q_R_init_before_grasp)
        

    ### INITIALIZE PLANNER
    def init_traj_planner(self):
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
        world_to_eef = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        eef_to_world = self.bc.invertTransform(world_to_eef[0], world_to_eef[1])
        world_to_cp = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]
        eef_to_cp = self.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                                world_to_cp[0], world_to_cp[1])

        # order for humanoid urdf: [yaw, pitch, roll]
        shoulder_min = [-3.1402077384521765, -0.248997453133789, -2.583238756496965]
        shoulder_max = [3.1415394736319917, 1.2392816988875348, -1.3229245882839409]
        # elbow_min = [0.401146]
        elbow_min = [0]
        elbow_max = [2.541304]  
        self.human_arm_lower_limits = shoulder_min + elbow_min
        self.human_arm_upper_limits = shoulder_max + elbow_max

        self.mppi_H_clamp = MPPI_H_Clamp(eef_to_cp, self.current_joint_angles, 
                                         self.human_arm_lower_limits, self.human_arm_upper_limits, self.human_rest_poses)

        # Instantiate trajectory planner
        self.trajectory_planner = TrajectoryPlanner(
            joint_limits=JOINT_LIMITS,
            robot_urdf_location=robot_urdf_location,
            scene_urdf_location=scene_urdf_location,
            link_fixed=LINK_FIXED,
            link_ee=LINK_EE,
            link_skeleton=LINK_SKELETON,
            control_points_location = control_points_location,
            control_points_number = 31,
            mppi_H_clamp = self.mppi_H_clamp,
        )
        print("Instantiated trajectory planner")

        # TODO update environment point cloud
        # self.robot_2.reset()
        # pcd = self.get_point_cloud()
        self.trajectory_planner.update_obstacle_pcd(self.pcd)
        print("Updated environment point cloud")

    def init_mppi_planner(self, current_joint_angles, target_joint_angles, clamp_by_human):
        # MPPI parameters
        N_JOINTS = len(self.robot.arm_controllable_joints)
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

        # Plan trajectory
        start_time = time.time()
        trajectory = self.trajectory_planner.get_mppi_rollout(self.current_joint_angles)
        print("planning time : ", time.time()-start_time)
        return trajectory
    
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

        view_matrix = self.bc.computeViewMatrix(
            cameraEyePosition= camera_eye_position,
            cameraTargetPosition=camera_look_at,
            cameraUpVector = [0,1,0],
            physicsClientId=self.bc._client
        )

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

    def get_point_cloud(self):
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
        far = 5
        near = 0.1

        ### Camera intrinsic:
        opengl_projection_matrix = (f_x/width,            0,                      0,                      0,
                                    skew/width,           f_y/height,           0,                      0,
                                    2*(c_x+0.5)/width-1,  2*(c_y+0.5)/height-1,   -(far+near)/(far-near), -1,
                                    0,                    0,                      -2*far*near/(far-near), 0)

        obj_pb_id = self.humanoid._humanoid
        obj_link_id = self.right_elbow

        # aabb = p.getAABB(obj_pb_id, obj_link_id, physicsClientId=self.bc._client)
        aabb = p.getAABB(obj_pb_id, physicsClientId=self.bc._client)
        aabb_min = np.array(aabb[0])
        aabb_max = np.array(aabb[1])
        obj_center = list((aabb_max + aabb_min)/2)

        ### Initialize point cloud and its normals
        pc = []
        normals = []

        camera_look_at = obj_center

        # phis = [45, 90, 135, 179]
        # thetas = [15, 45]
        phis = [30, 90, 130]
        thetas = [15, 45]

        for phi_deg in phis:
            for theta_deg in thetas:
                camera_distance = 3

                phi = np.deg2rad(phi_deg)
                theta = np.deg2rad(theta_deg)
                camera_eye_position = []
                camera_eye_position.append(camera_distance*np.cos(phi)*np.sin(theta) + obj_center[0])
                camera_eye_position.append(camera_distance*np.sin(phi)*np.sin(theta) + obj_center[1])
                camera_eye_position.append(camera_distance*np.cos(theta) + obj_center[2])

                view_matrix = self.bc.computeViewMatrix(
                    cameraEyePosition= camera_eye_position,
                    cameraTargetPosition=camera_look_at,
                    cameraUpVector = [0,1,0],
                    physicsClientId=self.bc._client
                )

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
                depth = far * near / (far - (far - near) * depth_buffer)
                obj_segmentation = seg > 0
                depth_buffer_seg = depth_buffer * obj_segmentation
                depth_seg = depth*obj_segmentation

                opengl_projection_matrix_np = np.transpose(np.reshape(np.array(opengl_projection_matrix), (4,4)))
                view_matrix_inv_np = np.transpose(np.reshape(np.array(view_matrix), (4,4))) # From world to camera, needs to be transposed because OpenGL is column-based
                view_matrix_np = np.linalg.inv(view_matrix_inv_np) # From camera to world

                pc_view = []            

                ### Generate point cloud
                for u in range(height):
                    for v in range(width):
                        if obj_segmentation[u,v]:
                            point_pixel = np.array([(2*v/width-1),(-2*u/height+1),2*depth_buffer[u,v]-1,1])
                            point_camera = np.matmul(np.linalg.inv(opengl_projection_matrix_np),point_pixel)
                            point_world = np.matmul(view_matrix_np,point_camera)
                            point_world /= point_world[3]
                            pc_view.append(point_world[:3])
                            pc.append(point_world[:3])

                pc_ply = o3d.geometry.PointCloud()
                pc_ply.points = o3d.utility.Vector3dVector(pc_view)

                pc_ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=300))        
                print("Print the normal vectors of the first 10 points")
                print(np.asarray(pc_ply.normals)[:10, :])

                pc_ply.orient_normals_towards_camera_location(camera_eye_position)
                normal = np.reshape(np.array(pc_ply.normals),np.shape(pc_view))

                pc_ply.clear()
                normal = normal.tolist()
                normals.extend(normal)

        ### Point cloud with normals per point
        pc_normals = np.concatenate((pc, normals),axis=1)
        dict = {'xyz': pc_normals}

        # np.save("pc_human_scene_rwe.npy", normal)

        ### Visualize point cloud
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pc)
        pc_ply.normals = o3d.utility.Vector3dVector(normals)
        # o3d.io.write_point_cloud("pc_human_scene_rwe.pcd", pc_ply)

        # cloud = o3d.io.read_point_cloud("pc_human_scene_rwe.pcd")
        o3d.visualization.draw_geometries([pc_ply])
        
        return np.array(pc_ply.points)


    ### HELPER FUNCTIONS
    def attach_human_arm_to_eef(self, joint_type=p.JOINT_FIXED):
        # attach human arm (obj) to eef (body)
        body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)              # world to eef 
        obj_pose = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)      # world to cp
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])             # eef to world
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],                      # eef to cp
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])
        self.eef_to_cp = obj_to_body
        
        self.cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                            parentLinkIndex=self.robot.eef_id,
                            childBodyUniqueId=self.humanoid._humanoid,
                            childLinkIndex=self.right_elbow,
                            jointType=joint_type,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))
    
    def deattach_human_arm_from_eef(self):
        self.bc.removeConstraint(self.cid)

    def draw_sphere_marker(self, position, radius=0.07, color=[1, 0, 0, 1]):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        self.marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

    def human_motion_from_frame_data(self, humanoid, motion, utNum):
        keyFrameDuration = motion.KeyFrameDuraction()
        self.bc.stepSimulation()
        humanoid.RenderReference(utNum * keyFrameDuration, self.bc)
        self.bc.stepSimulation()

    def move_robot(self, q_robot):
        for _ in range(500):
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, q_robot[i],
                                                force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
            self.bc.stepSimulation()

if __name__ == '__main__':
    env = HumanDemo()
    env.init_traj_planner()

    traj = env.init_mppi_planner(env.current_joint_angles, env.target_joint_angles, clamp_by_human=False)
    print(traj)

    for q in traj:
        for _ in range (300):
            for i, joint_id in enumerate(env.robot.arm_controllable_joints):
                env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
                                            force=500)
            env.bc.stepSimulation()
        time.sleep(0.5)

    for i, joint_id in enumerate(env.robot.arm_controllable_joints):
        print(i, env.bc.getJointState(env.robot.id, joint_id)[0])



    # # Step 1: trajectory before grasping
    # traj = env.init_mppi_planner(env.q_R_init_before_grasp, env.q_R_goal_before_grasp, clamp_by_human=False)
    # print(traj)

    # for q in traj:
    #     for _ in range (300):
    #         for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #             env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
    #                                         force=500)
    #         env.bc.stepSimulation()
    #     time.sleep(0.5)

    # for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #     print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

    # # Step 2: attach human arm to eef
    # env.attach_human_arm_to_eef()

    # # Step 3: trajectory after grasping
    # traj = env.init_mppi_planner(env.q_R_init_after_grasp, env.q_R_goal_after_grasp, clamp_by_human=True)
    # print(traj)

    # for q in traj:
    #     for _ in range (300):
    #         for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #             env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
    #                                         force=500)
    #         env.bc.stepSimulation()
    #     time.sleep(0.5)

    # for i, joint_id in enumerate(env.robot.arm_controllable_joints):
    #     print(i, env.bc.getJointState(env.robot.id, joint_id)[0])