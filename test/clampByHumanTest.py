import os, inspect
import os.path as osp
import pybullet as p
import math
import time
import sys
sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# robot
sys.path.append("/home/exx/Yubin/deep_mimic")
sys.path.append("/home/exx/Yubin/deep_mimic/mocap")
import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85
from pybullet_utils.bullet_client import BulletClient

# humanoid
from env.motion_capture_data import MotionCaptureData
from humanoid import Humanoid
from humanoid import HumanoidPose


def Reset(humanoid):
    global simTime
    humanoid.Reset()
    simTime = 0
    humanoid.SetSimTime(simTime)
    pose = humanoid.InitializePoseFromMotionData()
    humanoid.ApplyPose(pose, True, True, humanoid._humanoid, bc)

def draw_sphere_marker(bc, position, radius, color):
    vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    marker_id = bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

def violate_human_arm_limits(bc_second, robot, q_R, eef_to_cp, human_arm_lower_limits, human_arm_upper_limits):
    # get eef pose from q_R
    for i, joint in enumerate(robot.arm_controllable_joints):
        bc_second.setJointMotorControl2(robot.id, joint, p.POSITION_CONTROL, q_R[i])
        bc_second.stepSimulation()
    eef_to_world = bc_second.getLinkState(robot.id, robot.eef_id)[:2]

    # get cp pose
    cp_to_eef = bc_second.invertTransform(eef_to_cp[0], eef_to_cp[1])
    cp_to_world = bc_second.multiplyTransforms(cp_to_eef[0], cp_to_eef[1],
                                                    eef_to_world[0], eef_to_world[1])

    # IK -> get human joint angles
    rightShoulder = 3
    rightElbow = 4
    q_H = bc_second.calculateInverseKinematics(humanoid._humanoid, rightElbow, targetPosition=cp_to_world[0], targetOrientation=cp_to_world[1])

    # check q_H with joint limits, clamp
    clamped_q_H = []
    for i in range(len(q_H)):
        if q_H[i] < human_arm_lower_limits[i]:
            return True
        elif q_H[i] > human_arm_upper_limits[i]:
            return True
    
    return False

    # move humanoid in the 2nd server, get new cp pose
    bc_second.setJointMotorControlMultiDof(humanoid._humanoid, rightShoulder, controlMode=p.POSITION_CONTROL, targetPosition=clamped_q_H[:3])
    bc_second.setJointMotorControl2(humanoid._humanoid, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=clamped_q_H[3])
    bc_second.stepSimulation()
    cp_to_world = bc_second.getLinkState(humanoid._humanoid, rightElbow)[:2]

    # get new eef pose
    eef_to_world = bc_second.multiplyTransforms(eef_to_cp[0], eef_to_cp[1],
                                                    cp_to_world[0], cp_to_world[1])

    # IK -> get new robot joint angles
    q_R = bc_second.calculateInverseKinematics(robot.id, robot.eef_id, eef_to_world[0], eef_to_world[1],
                                                robot.arm_lower_limits, robot.arm_upper_limits, robot.arm_joint_ranges, robot.arm_rest_poses,
                                                maxNumIterations=20)
    q_R = [q_R[i] for i in range(len(robot.arm_controllable_joints))]
    
    return q_R
       

dataset_path = 'data/data_3d_h36m.npz'
motionPath = 'data/Sitting1.json'
json_path = 'data/Sitting1.json'
subject = 'S11'
action = 'Sitting1'
fps = 24
loop = 'wrap'


# load environment
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, -9.8, 0) 

y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor
bedID = bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed
table1ID = bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table
table2ID = bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table
block_id = bc.loadURDF("cube.urdf", (-1.0, 0.15, 0), y2zOrn, useFixedBase=True, globalScaling=0.45)

# load robot
robot = UR5Robotiq85(bc, (-1.0, 0.35, 0), (-1.57, 0, 0))
robot.load()
robot.reset()

# load humanoid
motion = MotionCaptureData()
motion.Load(motionPath)
humanoid = Humanoid(bc, motion, [0, 0.3, 0])

# 2nd BC server
bc_second = BulletClient(connection_mode=p.DIRECT)
bc_second.setAdditionalSearchPath(pybullet_data.getDataPath())
bc_second.configureDebugVisualizer(bc_second.COV_ENABLE_Y_AXIS_UP, 1)

# load humanoid in the 2nd server
motionPath = 'data/Greeting.json'
motion = MotionCaptureData()
motion.Load(motionPath)
humanoid = Humanoid(bc_second, motion, [0, 0.3, 0])

# TODO robot joint angles before clamp
q_R_init = [-0.965373317701583, -1.0580729514764646, 1.569563232784218, -1.9578727953619697, -1.4855413446137167, -0.9699865449395163]
q_R_goal = [-0.11242203, -1.27380349,  1.06544561, -1.35751674, -1.42131799, -0.11199235]

# load robot in the 2nd server
robot2 = UR5Robotiq85(bc_second, (-1.0, 0.35, 0), (-1.57, 0, 0))
robot2.load()
robot2.reset()

# move robot to q_R_init
for i, joint_id in enumerate(robot.arm_controllable_joints):
    bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_R_init[i])
    bc.stepSimulation()
for i, joint_id in enumerate(robot2.arm_controllable_joints):
    bc_second.setJointMotorControl2(robot2.id, joint_id, p.POSITION_CONTROL, q_R_init[i])
    bc_second.stepSimulation()

# TODO robot joint angles after clamp
eef_to_world = bc.getLinkState(robot.id, robot.eef_id)[:2]
cp_to_world = bc.getLinkState(humanoid._humanoid, 4)[:2]
world_to_cp = bc.invertTransform(cp_to_world[0], cp_to_world[1])
eef_to_cp = bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                world_to_cp[0], world_to_cp[1])

shoulder_min = bc.getEulerFromQuaternion([-0.397618, -0.885139, -0.759182, -0.014004])
shoulder_max = bc.getEulerFromQuaternion([0.654265, 0.959910, 0.718185, 0.643939])
elbow_min = [0.4]
elbow_max = [2.5]
human_arm_lower_limits = shoulder_min + shoulder_max
human_arm_upper_limits = elbow_min + elbow_max

q_R_clamped = clamp_human_joints(bc_second, robot2, q_R)
for _ in range(50):
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_R[i])
        bc.stepSimulation()
print('Done!')
    

# while True:
#     p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

#     # bc.setJointMotorControl2(robot.id, 1, p.POSITION_CONTROL, 1.0)
#     robot.reset()

#     # robot.move_ee(action=parameter[:-1], control_method='end')
#     # robot.move_gripper(parameter[-1])

#     p.stepSimulation()


Reset(humanoid)
p.disconnect()

