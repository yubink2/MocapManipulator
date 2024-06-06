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

# ramp
from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower

# mppi H clamp
from mppi_planning.mppi_human_clamping_rwe import MPPI_H_Clamp

# point cloud
import open3d as o3d


class HumanDemo():
    def __init__(self):
        self.bc = BulletClient(connection_mode=p.GUI)
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.bc.configureDebugVisualizer(self.bc.COV_ENABLE_Y_AXIS_UP, 1)
        self.bc.setGravity(0, -9.8, 0) 
        self.bc.setTimestep = 0.0005

        y2zOrn = self.bc.getQuaternionFromEuler((-1.57, 0, 0))

        # load environment
        plane_id = self.bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)

        # load human
        baseOrn = self.bc.getQuaternionFromEuler((1.57, 0, 1.57))
        self.humanoid = self.bc.loadURDF("urdf/humanoid_for_rwe.urdf", (0, 0.08, 0), baseOrn, 
                                         globalScaling=0.33, useFixedBase=True)
        self.left_shoulder = 6
        self.left_elbow = 7
        self.human_rest_poses = [0.0, 0.0]

        for j in range(self.bc.getNumJoints(self.humanoid)):  # change colors of the human model limbs
            ji = self.bc.getJointInfo(self.humanoid, j)
            self.bc.changeDynamics(self.humanoid, j, linearDamping=0, angularDamping=0)
            self.bc.changeVisualShape(self.humanoid, j, rgbaColor=[1, 1, 1, 1])

        for i in range(self.bc.getNumJoints(self.humanoid)):
            print(self.bc.getJointInfo(self.humanoid, i))

        # load robot
        self.robot = UR5Robotiq85(self.bc, (-0.95, 0.53, 0.9), (-1.57, 0, 0), globalScaling=1.0)
        self.robot.load()
        self.robot.reset()

        # compute init_after_grasp and goal_after_grasp 
        self.robot.open_gripper()
        self.init_robot_configs()
        self.set_robot_target_joint()

    def init_robot_configs(self):
        # initial human arm
        self.init_human_left_arm = [3.14, 0.0]
        self.move_human_arm(self.init_human_left_arm)

        self.robot.reset()

        # # DEUBGGING
        # self.ur5_debug_parameter()

        # move robot to grasp pose
        pos = self.bc.getLinkState(self.humanoid, self.left_elbow)[0]
        pos_up_2 = [pos[0], pos[1]+0.4, pos[2]]
        orn_2 = [0, 0, -1.57]
        pos_up_1 = [pos[0], pos[1]+0.17, pos[2]]
        orn_1 = [0, 0, -1.57]

        for _ in range(200):
            self.robot.move_ee(action=pos_up_2+orn_2, control_method='end')
            self.bc.stepSimulation()

        # save joint angles --> q_R_init_before_grasp
        q_R_init_before_grasp = []
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            q_R_init_before_grasp.append(self.bc.getJointState(self.robot.id, joint_id)[0])
        self.q_R_init_before_grasp = q_R_init_before_grasp
        print('q_R_init_before_grasp', self.q_R_init_before_grasp)

        for _ in range(100):
            self.robot.move_ee(action=pos_up_1+orn_1, control_method='end')
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
        body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)   # world to eef 
        obj_pose = self.bc.getLinkState(self.humanoid, self.left_elbow)      # world to cp
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])  # eef to world
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],           # eef to cp
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])
        self.eef_to_cp = obj_to_body
        
    def attach_human_arm_to_eef(self, joint_type=p.JOINT_FIXED):
        # attach human arm (obj) to eef (body)
        body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)   # world to eef 
        obj_pose = self.bc.getLinkState(self.humanoid, self.left_elbow)      # world to cp
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])  # eef to world
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],           # eef to cp
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])
        self.eef_to_cp = obj_to_body
        
        self.cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                            parentLinkIndex=self.robot.eef_id,
                            childBodyUniqueId=self.humanoid,
                            childLinkIndex=self.left_elbow,
                            jointType=joint_type,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))
    
    def deattach_human_arm_from_eef(self):
        self.bc.removeConstraint(self.cid)

    def set_robot_target_joint(self):
        self.attach_human_arm_to_eef()

        # move human to target config
        self.target_human_left_arm = [1.8, 0.6]
        self.move_human_arm(self.target_human_left_arm)

        print(self.bc.getJointState(self.humanoid, self.left_elbow)[0])

        # save joint angles --> q_R_goal_before_grasp
        q_R_goal_after_grasp = []
        for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            q_R_goal_after_grasp.append(self.bc.getJointState(self.robot.id, joint_id)[0])
        self.q_R_goal_after_grasp = q_R_goal_after_grasp
        print('q_R_goal_after_grasp', self.q_R_goal_after_grasp)

        eef_goal_pos = self.bc.getLinkState(self.robot.id, self.robot.eef_id)[:2]

        self.deattach_human_arm_from_eef()

        # move human back to init config
        self.move_human_arm(self.init_human_left_arm)

        # move robot to back to init config
        self.move_robot(self.q_R_init_before_grasp)

        # ###### DEBUGGING

        # # move robot to target config
        # for _ in range(500):
        #     for i, joint_id in enumerate(self.robot.arm_controllable_joints):
        #         self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, q_R_goal_after_grasp[i],
        #                                         force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
        #     self.bc.stepSimulation()

        # # move human to target config
        # for _ in range(500):
        #     self.bc.setJointMotorControl2(self.humanoid, self.left_shoulder, p.POSITION_CONTROL,
        #                                   targetPosition=self.target_human_left_arm[0], positionGain=0.03, force=500)
        #     self.bc.setJointMotorControl2(self.humanoid, self.left_elbow, p.POSITION_CONTROL,
        #                                   targetPosition=self.target_human_left_arm[1], positionGain=0.03, force=500)
        #     self.bc.stepSimulation()

        # while True:
        #     self.bc.stepSimulation()

        # ###### DEBUGGING

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

        obj_pb_id = self.humanoid
        obj_link_id = self.left_elbow

        aabb = p.getAABB(obj_pb_id, obj_link_id, physicsClientId=self.bc._client)
        # aabb = p.getAABB(obj_pb_id, physicsClientId=self.bc._client)
        aabb_min = np.array(aabb[0])
        aabb_max = np.array(aabb[1])
        obj_center = list((aabb_max + aabb_min)/2)

        camera_look_at = obj_center

        phi_deg = 130
        theta_deg = 15
        camera_distance = 1

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

        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pc_view)

        pc_ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=300))        
        print("Print the normal vectors of the first 10 points")
        print(np.asarray(pc_ply.normals)[:10, :])

        pc_ply.orient_normals_towards_camera_location(camera_eye_position)
        normal = np.reshape(np.array(pc_ply.normals),np.shape(pc_view))

        # np.save("pc_human_scene_rwe.npy", normal)

        ### Visualize point cloud
        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pc_view)
        pc_ply.normals = o3d.utility.Vector3dVector(normal)
        # o3d.io.write_point_cloud("pc_human_scene_rwe.pcd", pc_ply)

        # cloud = o3d.io.read_point_cloud("pc_human_scene_rwe.pcd")
        o3d.visualization.draw_geometries([pc_ply])
        
        return np.array(pc_ply.points)

    def init_traj_planner(self):
        # urdf paths
        robot_urdf_location = 'pybullet_ur5/urdf/ur5_robotiq_85.urdf'
        scene_urdf_location = 'resources/environment/environment.urdf'
        control_points_location = 'resources/ur5_control_points/control_points_rwe.json'

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

        # instantiate mppi H clamp
        self.human_arm_lower_limits = [0.0, 0.0]
        self.human_arm_upper_limits = [3.14, 1.57]
        # self.mppi_H_clamp = MPPI_H_Clamp(self.eef_to_cp, self.human_arm_lower_limits, self.human_arm_upper_limits, self.human_rest_poses)
        self.mppi_H_clamp = MPPI_H_Clamp(self.eef_to_cp, self.human_arm_lower_limits, self.human_arm_upper_limits, self.target_human_left_arm)

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
            clamp_by_human = True,
        )
        print("Instantiated trajectory planner")

        # Update env point cloud
        pcd = self.get_camera_img()
        self.trajectory_planner.update_obstacle_pcd(pcd)
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
    
    def draw_sphere_marker(self, position, radius=0.07, color=[1, 0, 0, 1]):
        vs_id = self.bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        self.marker_id = self.bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

    def ur5_debug_parameter(self):
        pos = self.bc.getLinkState(self.humanoid, self.left_elbow)[0]
        pos_up_2 = (pos[0], pos[1]+0.15, pos[2]-0.09)
        pos_up = (pos[0], pos[1]+0.17, pos[2])
        print(pos)

        position_control_group = []
        position_control_group.append(p.addUserDebugParameter('x', -1.5, 1.5, pos_up_2[0]))
        position_control_group.append(p.addUserDebugParameter('y', -1.5, 1.5, pos_up_2[1]))
        position_control_group.append(p.addUserDebugParameter('z', -1.5, 1.5, pos_up_2[2]))
        position_control_group.append(p.addUserDebugParameter('roll', -3.14, 3.14, 0))
        position_control_group.append(p.addUserDebugParameter('pitch', -3.14, 3.14, -0.56))
        position_control_group.append(p.addUserDebugParameter('yaw', -3.14, 3.14, -1.57))
        position_control_group.append(p.addUserDebugParameter('gripper_opening', 0, 0.085, 0.08))

        while True:
            self.bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

            parameter = []
            for i in range(len(position_control_group)):
                parameter.append(self.bc.readUserDebugParameter(position_control_group[i]))

            self.robot.move_ee(action=parameter[:-1], control_method='end')
            self.robot.move_gripper(parameter[-1])

            self.bc.stepSimulation()

            # for i, joint_id in enumerate(self.robot.arm_controllable_joints):
            #     print(i, self.bc.getJointState(self.robot.id, joint_id)[0])

            # print(self.left_shoulder, self.bc.getJointState(self.humanoid, self.left_shoulder)[0])
            # print(self.left_elbow, self.bc.getJointState(self.humanoid, self.left_elbow)[0])

    def move_human_arm(self, q_human_left_arm):
        for _ in range(500):
            self.bc.setJointMotorControl2(self.humanoid, self.left_shoulder, p.POSITION_CONTROL,
                                          targetPosition=q_human_left_arm[0], positionGain=0.03, force=500)
            self.bc.setJointMotorControl2(self.humanoid, self.left_elbow, p.POSITION_CONTROL,
                                          targetPosition=q_human_left_arm[1], positionGain=0.03, force=500)
            self.bc.stepSimulation()

    def move_robot(self, q_robot):
        for _ in range(500):
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                self.bc.setJointMotorControl2(self.robot.id, joint_id, p.POSITION_CONTROL, q_robot[i],
                                                force=self.robot.joints[joint_id].maxForce, maxVelocity=self.robot.joints[joint_id].maxVelocity)
            self.bc.stepSimulation()

if __name__ == '__main__':
    env = HumanDemo()

    ####### TESTIG TRAJECTORY
    # Step 0: initialize human arm & robot
    dir = "rwe_trajs/FINAL_human_on_ground_robot_up/"
    dir = ""
    trial = 3
    traj = np.load(dir + f"traj_before_grasping_{trial}.npy")
    print(traj)

    init_human_left_arm = [3.14, 0.0]
    init_robot = traj[0]
    env.move_human_arm(init_human_left_arm)
    env.move_robot(init_robot)

    # Step 1: trajectory before grasping
    for q in traj:
        for _ in range (300):
            for i, joint_id in enumerate(env.robot.arm_controllable_joints):
                env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
                                            force=500)
            env.bc.stepSimulation()
        time.sleep(0.5)

    for i, joint_id in enumerate(env.robot.arm_controllable_joints):
        print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

    # Step 2: attach human arm to eef
    env.attach_human_arm_to_eef(joint_type=p.JOINT_POINT2POINT)
    # env.attach_human_arm_to_eef()
    time.sleep(2)

    # Step 3:trajectory after grasping
    traj = np.load(dir + f"traj_after_grasping_{trial}.npy")
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

    time.sleep(10)
    sys.exit()


    ####### GETTIGN NEW TRAJECTORY
    trial = 4
    env.init_traj_planner()

    # Step 1: trajectory before grasping
    traj = env.init_mppi_planner(env.q_R_init_before_grasp, env.q_R_goal_before_grasp, clamp_by_human=False)
    print(traj)
    np.save(f"traj_before_grasping_{trial}.npy", traj)

    for q in traj:
        for _ in range (300):
            for i, joint_id in enumerate(env.robot.arm_controllable_joints):
                env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
                                            force=500)
            env.bc.stepSimulation()
        time.sleep(0.5)

    for i, joint_id in enumerate(env.robot.arm_controllable_joints):
        print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

    # # Step 2: attach human arm to eef
    # env.attach_human_arm_to_eef(joint_type=p.JOINT_POINT2POINT)

    # Step 3: trajectory after grasping
    traj = env.init_mppi_planner(env.q_R_init_after_grasp, env.q_R_goal_after_grasp, clamp_by_human=True)
    print(traj)
    np.save(f"traj_after_grasping_{trial}.npy", traj)

    for q in traj:
        for _ in range (300):
            for i, joint_id in enumerate(env.robot.arm_controllable_joints):
                env.bc.setJointMotorControl2(env.robot.id, joint_id, p.POSITION_CONTROL, q[i],
                                            force=500)
            env.bc.stepSimulation()
        time.sleep(0.5)

    for i, joint_id in enumerate(env.robot.arm_controllable_joints):
        print(i, env.bc.getJointState(env.robot.id, joint_id)[0])





### NOTE
# first few initial waypoints for traj_after_grasping are pushing the arm down..? need to check!

# q_init = robot reset pose (q_R_init_before_grasp) --> pos_up_2
# .. (traj)
# q_goal1 = robot graps pose (q_R_goal_before_grasp) (q_R_init_after_grasp) --> pos_up
# attach human arm to eef
# .. (traj)
# q_goal2 = robot goal config (q_R_goal_after_grasp)
