"""Credit to: https://github.com/lyfkyle/pybullet_ompl/"""

# ur5 ompl
import os, inspect
import os.path as osp
import pybullet as p
import math
import sys
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("pybullet_ompl")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet_data
import pb_ompl_ur5
from models.robot import UR5Robotiq85
from pybullet_utils.bullet_client import BulletClient
import time

# humanoid
from deep_mimic.env.motion_capture_data import MotionCaptureData
from humanoid import Humanoid
from humanoid import HumanoidPose

# CMP
sys.path.append("constraint")
from tsr import *
from ConstrainedPlanningCommon import *
import numpy as np
from transformation import *


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

        # add obstacles
        self.obstacles.append(plane_id)
        self.obstacles.append(bed_id)
        self.obstacles.append(table1_id)
        self.obstacles.append(table2_id)
        # self.obstacles.append(block_id)

        # load robot
        self.robot = UR5Robotiq85(self.bc, (-1.0, 0.35, 0), (-1.57, 0, 0))
        self.robot.load()
        self.robot.reset()
        self.init_robot_configs()

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl_ur5.PbOMPL(self.robot, self.obstacles)
        self.pb_ompl_interface.set_planner("BITstar")

        # add obstacles
        self.add_obstacles()

    def init_robot_configs(self):
        # move robot to grasp pose
        pos, orn = self.bc.getLinkState(self.humanoid._humanoid, 4)[:2]  
        pos_up = (pos[0], pos[1]+0.13, pos[2])
        for _ in range(50):
            self.robot.move_ee(action=pos_up + (-1.57, 0.15, -1.57), control_method='end')
            self.bc.stepSimulation()
        
        # attach human arm to eef
        body_pose = self.bc.getLinkState(self.robot.id, self.robot.eef_id)
        obj_pose = self.bc.getLinkState(self.humanoid._humanoid, 4)
        world_to_body = self.bc.invertTransform(body_pose[0], body_pose[1])
        obj_to_body = self.bc.multiplyTransforms(world_to_body[0],
                                            world_to_body[1],
                                            obj_pose[0], obj_pose[1])

        cid = self.bc.createConstraint(parentBodyUniqueId=self.robot.id,
                            parentLinkIndex=self.robot.eef_id,
                            childBodyUniqueId=self.humanoid._humanoid,
                            childLinkIndex=4,
                            jointType=p.JOINT_FIXED,
                            jointAxis=(0, 0, 0),
                            parentFramePosition=obj_to_body[0],
                            parentFrameOrientation=obj_to_body[1],
                            childFramePosition=(0, 0, 0),
                            childFrameOrientation=(0, 0, 0))

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)

    def add_obstacles(self):
        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def demo(self):
        # start = [0, 0, 0, -1, 0, 1.5, 0]
        # goal = [0, 1.5, 0, -0.1, 0, 0.2, 0]
        # start = [0.2, 0, 0, -1, 0, 1.5]
        # goal = [-1, -0.5, -0.5, -0.1, 0, 0.2]
        start = env.robot.get_q()
        goal = [-1, -1, 1, -2, -1.5, -0.5]

        self.robot.set_state(start)
        res, path = self.pb_ompl_interface.plan(goal)
        if res:
            self.pb_ompl_interface.execute(path)
        return res, path

def pose_to_transform(pos, orn):
    translation_matrix = np.array([
        [1, 0, 0, pos[0]],
        [0, 1, 0, pos[1]],
        [0, 0, 1, pos[2]],
        [0, 0, 0, 1]
    ])

    rotation_matrix = quaternion_matrix(orn)

    transform_matrix = np.dot(translation_matrix, rotation_matrix)
    return transform_matrix

def transform_to_pose(transform_matrix):
    pos = transform_matrix[:3, 3]
    orn = quaternion_from_matrix(transform_matrix[:3, :3])
    return pos, orn

def define_tsrchain(env):
    # rotation of arm about its elbow
    human_pos, human_orn = env.bc.getLinkState(env.humanoid._humanoid, 4)[4:6]   # elbow pose in world frame
    T0_w = pose_to_transform(human_pos, human_orn)
    Tw_e = np.eye(4)
    Tw_e[1][3] = -0.12
    Bw = np.zeros((6,2))
    constraint1 = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw)

    # rotation of eef about grasp pose
    eef_pos, eef_orn = env.bc.getLinkState(env.robot.id, env.robot.eef_id)[4:6]  # eef pose in world frame
    pos_up = (human_pos[0], human_pos[1]+0.13, human_pos[2])
    T0_w = pose_to_transform(pos_up, env.bc.getQuaternionFromEuler([-1.57, 0.15, -1.57])) 
    # T0_w = pose_to_transform(pos_up, human_orn) 
    T0_e = pose_to_transform(eef_pos, eef_orn)
    Tw_e = np.linalg.inv(T0_w) @ T0_e  # desire pose of eef relative to grasp
    T0_w = np.eye(4)
    Bw = np.zeros((6,2))
    # Bw[4][:] = [-2.5, -0.4]
    # Bw[3][:] = [0.2, 2.5]
    # Bw[4][:] = [0.4, 2.5]
    Bw[5][:] = [1.0, 2.5]
    constraint2 = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw)

    # constraint applied over the whole trajectory
    tsrchain = TSRChain(sample_start=False, sample_goal=False, constrain=True, 
                                      TSRs = [constraint1, constraint2])
    return tsrchain

if __name__ == '__main__':
    env = HumanDemo()

    tsrchain = define_tsrchain(env)
    # for _ in range (10):
    #     print(tsrchain.sample())  # sample pose of the end-effector
    sample = tsrchain.sample()
    pos, orn = transform_to_pose(sample)
    print(sample)
    print(pos, orn)
    print(np.concatenate((pos, orn)))
    time.sleep(8)

    while (True):
        env.bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        env.robot.move_ee(action=np.concatenate((pos, env.bc.getEulerFromQuaternion(orn))), control_method='end')
        env.bc.stepSimulation()

    # env.demo()




# elbow range: [0.401146, 2.541233]