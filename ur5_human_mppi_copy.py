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

# informed rrt star
from informed_rrtstar.informed_rrtstar_3d import InformedRRTStar


class HumanDemo():
    def __init__(self):
        self.obstacles = []

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
        table1_id = self.bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.3), y2zOrn, globalScaling=0.6)  # table
        table2_id = self.bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.3), y2zOrn, globalScaling=0.6)  # table

        # load human
        motionPath = 'data/Sitting1.json'
        motion = MotionCaptureData()
        motion.Load(motionPath)
        self.humanoid = Humanoid(self.bc, motion, [0, 0.3, 0])
        self.right_shoulder_y = 3
        self.right_shoulder_p = 4
        self.right_shoulder_r = 5
        self.right_elbow = 7
        self.human_rest_poses = [2.4790802489002552, -0.01642306738465106, -1.8128412472566666, 0.4529190452054409]
        self.human_motion_from_frame_data(self.humanoid, motion, 230)

        # add obstacles
        self.obstacles.append(plane_id)
        self.obstacles.append(bed_id)
        self.obstacles.append(table1_id)
        self.obstacles.append(table2_id)

        # TODO get camera image
        self.get_camera_img()

        # while True:
        #     self.robot.reset()
        #     self.bc.stepSimulation()

        # load robot
        self.robot = UR5Robotiq85(self.bc, (-0.75, 0, 0), (-1.57, 0, 0), globalScaling=1.2)
        self.robot.load()
        self.robot.reset()

        # TODO load second robot

        # TODO call inference.py to get grasps

        # initialize robot config
        self.init_robot_configs()
        self.set_robot_target_joint(motion, frame=230)

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

    def init_robot_configs(self):
        # move robot to grasp pose
        pos, orn = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)[:2]

        # KWONJI
        pos_up_2 = (-0.17962980178173082, 0.7, 0.1)
        pos_up = (-0.17962980178173082, 0.590458026205689, 0.3667859122611105)
        orn = (0.42581023056381473, 0.025895246484703916, 0.8784134154854197, -0.21541809406808593)

        if pos_up_2 is not None:
            current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, pos_up_2, orn,
                                                       self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                       maxNumIterations=20)

            for _ in range (100):
                for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                    self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, current_joint_angles[i],
                                                    force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
                self.bc.stepSimulation()
        

        current_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, pos_up, orn,
                                                       self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                                       maxNumIterations=20)
        self.current_joint_angles = [current_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]

        for _ in range (100):
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, current_joint_angles[i],
                                                force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
            self.bc.stepSimulation()
        print('moved robot to init config')

        print('current_joint_angles', self.current_joint_angles)
                                                 

        # attach human arm (obj) to eef (body)
        body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)  # world to eef 
        obj_pose = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)  # world to cp
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])  # eef to world
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],          # eef to cp
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])
        self.eef_to_cp = obj_to_body
        
        cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                            parentLinkIndex=self.robot.eef_id,
                            childBodyUniqueId=self.humanoid._humanoid,
                            childLinkIndex=self.right_elbow,
                            jointType=p.JOINT_FIXED,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))

    def set_robot_target_joint(self, motion, frame):
        # move human to target config
        self.human_motion_from_frame_data(self.humanoid, motion, frame)
        world_to_cp = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow)
        cp_to_eef = self.bc.invertTransform(self.eef_to_cp[0], self.eef_to_cp[1])
        world_to_eef = self.bc.multiplyTransforms(world_to_cp[0], world_to_cp[1],
                                                  cp_to_eef[0], cp_to_eef[1])
        
        # get robot joint config
        target_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                            self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
                                            maxNumIterations=20)
        target_joint_angles = [target_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
        self.target_joint_angles = target_joint_angles
        print('target_joint_angles', target_joint_angles)

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def draw_sphere_marker(self, position, radius=0.07, color=[1, 0, 0, 1]):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        self.marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

    def draw_frame(self, position, quaternion=[0, 0, 0, 1]):
        m = R.from_quat(quaternion).as_matrix()
        x_vec = m[:, 0]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for color, column in zip(colors, range(3)):
            vec = m[:, column]
            from_p = position
            to_p = position + (vec * 0.1)
            p.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)

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
            control_points_number = 21,
            mppi_H_clamp = self.mppi_H_clamp,
        )
        print("Instantiated trajectory planner")

        # MPPI parameters
        N_JOINTS = len(self.robot.arm_controllable_joints)
        mppi_control_limits = [
            -0.05 * np.ones(N_JOINTS),
            0.05 * np.ones(N_JOINTS)
        ]
        mppi_nsamples = 500
        mppi_covariance = 0.005
        mppi_lambda = 1.0

        # robot goal pose
        # cp_pos, cp_orn = env.bc.getLinkState(env.humanoid._humanoid, self.right_elbow)[:2]
        # cp_pos = (cp_pos[0]+0.15, cp_pos[1]+0.25, cp_pos[2]+0.15)  # feasible goal config
        # cp_pos = (cp_pos[0]+0.1, cp_pos[1]+0.35, cp_pos[2]-0.1)  # feasible goal config
        cp_pos = [0.08, 0.38, 0.284]
        cp_orn = [-0.562, 0.727, -0.297] #env.bc.getQuaternionFromEuler((-0.562, 0.727, -0.297))
        grasp_pose = cp_pos + cp_orn
                  
        # KWONJI 1
        # cp_orn = env.bc.getQuaternionFromEuler((-1.57, 0.35, -1.75))

        # KWONJI 2
        # cp_orn = env.bc.getQuaternionFromEuler((-1.2, -0.6, -1.8))

        self.eef_goal_pose = (cp_pos, cp_orn)

        # # mark goal pose
        # self.draw_sphere_marker(cp_pos, radius=0.05, color=[0, 1, 0, 1])

        # while True:
        #     self.robot.move_ee(action=grasp_pose, control_method='end')
        #     self.bc.stepSimulation() 

        # world_to_cp_goal = ((-0.4968800171961967, 0.40857648299834726, 0.38046846967865205),
        #                     (0.917834969986322, 0.31931322730674455, 0.22248896777694108, -0.07820927026069255))
        # # cp_goal_to_world = self.bc.invertTransform(world_to_cp_goal[0], world_to_cp_goal[1])
        # cp_to_eef = self.bc.invertTransform(eef_to_cp[0], eef_to_cp[1])
        # world_to_eef_goal = self.bc.multiplyTransforms(world_to_cp_goal[0], world_to_cp_goal[1],
        #                                               cp_to_eef[0], cp_to_eef[1])

        # # mark goal pose
        # self.draw_sphere_marker(world_to_cp_goal[0], radius=0.05, color=[0, 1, 0, 1])
        # self.draw_sphere_marker(world_to_eef_goal[0], radius=0.05, color=[1,0,0,1])
        # while True:
        #     self.bc.stepSimulation()
        
        # # Find joint angles
        # current_joint_angles = self.current_joint_angles
        # target_joint_angles = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, cp_pos, cp_orn,
        #                                     self.robot.arm_lower_limits, self.robot.arm_upper_limits, self.robot.arm_joint_ranges, self.robot.arm_rest_poses,
        #                                     maxNumIterations=20)
        # target_joint_angles = [target_joint_angles[i] for i in range(len(self.robot.arm_controllable_joints))]
        # self.target_joint_angles = target_joint_angles
        # print("q_init: ", current_joint_angles)
        # print("q_goal: ", target_joint_angles)

        # TODO set joint angles
        current_joint_angles = self.current_joint_angles
        # target_joint_angles = [-1.831, -2.231, 1.914, -1.047, -0.110, -1.836]
        # target_joint_angles = [-1.238, -2.177, 2.047, -2.698, -0.723, -0.579]
        # target_joint_angles = [-1.663, -2.202, 2.015, -1.893, -0.485, -0.718]

        # target_joint_angles = [-0.891, -1.955, 1.735, -2.242, -0.862, 2.426]

        target_joint_angles = [-1.213, -1.369, 1.310, -1.774, -1.646, -0.115]
        self.target_joint_angles = target_joint_angles
        print("q_init: ", current_joint_angles)
        print("q_goal: ", target_joint_angles)

        # for _ in range(1000):
        #     for i, joint_id in enumerate(self.robot.arm_controllable_joints):
        #         self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, target_joint_angles[i],
        #                                       force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
        #     self.bc.stepSimulation() 
        
        # for i, joint_id in enumerate(self.robot.arm_controllable_joints):
        #     print(i, self.bc.getJointState(self.robot.id, joint_id)[0])
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

        # Plan trajectory
        start_time = time.time()
        trajectory = self.trajectory_planner.get_mppi_rollout(current_joint_angles)
        print("planning time : ", time.time()-start_time)
        return trajectory

    def init_informed_rrtstar_planner(self, traj):
        self.informed_rrtstar_planner = InformedRRTStar(self.current_joint_angles, self.target_joint_angles, self.obstacles,
                                                        self.robot.id, self.robot.arm_controllable_joints)
        return self.informed_rrtstar_planner.plan(traj)

    def test_clamping(self, q_R):
        if env.mppi_H_clamp.violate_human_arm_limits(q_R):
            print('violated')
        else:
            print('we good')
    
    def distance_to_goal(self):
        eef_world_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]
        return np.linalg.norm(np.array(eef_world_pose[0]) - np.array(self.eef_goal_pose[0]))
    
    def human_motion_from_frame_data(self, humanoid, motion, utNum):
        keyFrameDuration = motion.KeyFrameDuraction()
        self.bc.stepSimulation()
        humanoid.RenderReference(utNum * keyFrameDuration, self.bc)


if __name__ == '__main__':
    env = HumanDemo()

    traj = env.init_mppi_planner()
    print('MPPI planner done')

    # # Check if last waypoint is violated
    # if env.mppi_H_clamp.violate_human_arm_limits(traj[len(traj)-1]):
    #     print('last waypoint violated, removed.')
    #     traj = traj[:-1]

    print(traj)

    for q in traj:
        for _ in range (100):
            for i, joint_id in enumerate(env.robot.arm_controllable_joints):
                env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
                                                force=env.robot.joints[joint_id].maxForce, maxVelocity=env.robot.joints[joint_id].maxVelocity)
            env.bc.stepSimulation()
        time.sleep(0.5)

    for i, joint_id in enumerate(env.robot.arm_controllable_joints):
        print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

    time.sleep(5)

