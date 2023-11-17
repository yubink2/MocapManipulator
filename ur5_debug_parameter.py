import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import numpy as np
import pybullet as p
import pybullet_data

from tqdm import tqdm
from pybullet_ur5.robot import Panda, UR5Robotiq85, UR5Robotiq140
from pybullet_ur5.utilities import YCBModels, Camera
import time
import math
import numpy as np

from pybullet_utils.bullet_client import BulletClient
from deep_mimic.env.motion_capture_data import MotionCaptureData

from humanoid import Humanoid
from humanoid import HumanoidPose

from deepmimic_json_generator import *
from transformation import *


import sys
# sys.path
# ['', '/home/exx/anaconda3/envs/py37-ompl-test/lib/python37.zip', '/home/exx/anaconda3/envs/py37-ompl-test/lib/python3.7', '/home/exx/anaconda3/envs/py37-ompl-test/lib/python3.7/lib-dynload', '/home/exx/anaconda3/envs/py37-ompl-test/lib/python3.7/site-packages']
sys.path.append("/usr/lib/python3/dist-packages")
import ompl
# '/usr/lib/python3/dist-packages/ompl/__init__.py'
sys.path.append("/usr/lib/python3/dist-packages")



def Reset(humanoid):
  global simTime
  humanoid.Reset()
  simTime = 0
  humanoid.SetSimTime(simTime)
  pose = humanoid.InitializePoseFromMotionData()
  humanoid.ApplyPose(pose, True, True, humanoid._humanoid, bc)

def euler_angles_from_vector(position, center):
    if center[0] > position[0]:
        x,y,z = center-position
    else:
        x,y,z = position-center
        
    length = math.sqrt(x**2 + y**2 + z**2)
    pitch = math.acos(z/length)
    yaw = math.atan(y/x)
    roll = math.pi if position[0] > center[0] else 0

    euler_angles = [roll,pitch,yaw]
    return euler_angles

def human_motion_from_frame_data(humanoid, utNum, bc_arg):
  # print('bc: ', bc)
  keyFrameDuration = motion.KeyFrameDuraction()
  bc_arg.stepSimulation()
  humanoid.RenderReference(utNum * keyFrameDuration)  # RenderReference calls Slerp() & ApplyPose()

def human_motion_from_frame_data_without_applypose(humanoid, utNum, bc_arg):
  keyFrameDuration = motion.KeyFrameDuraction()
  bc_arg.stepSimulation()
  pose = humanoid.RenderReferenceWithoutApplyPose(utNum * keyFrameDuration)
  print('--human_zmotion_from_frame_data_without_applypose: ', pose._rightShoulderRot)
  print('--human_zmotion_from_frame_data_without_applypose: ', pose._rightElbowRot)


dataset_path = 'data/data_3d_h36m.npz'
motionPath = 'data/Sitting1.json'
json_path = 'data/Sitting1.json'
subject = 'S11'
action = 'Sitting1'
fps = 24
loop = 'wrap'

# dataset = init_fb_h36m_dataset(dataset_path)
# ground_truth = pose3D_from_fb_h36m(dataset, subject=subject, action=action, shift=[1.0, 0.0, 0.0])
# rot_seq = coord_seq_to_rot_seq(coord_seq=ground_truth, frame_duration=1 / fps)
# rot_seq_to_deepmimic_json(rot_seq=rot_seq, loop=loop, json_path=json_path)


bc = BulletClient(connection_mode=p.GUI)

# bc.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, -9.8, 0) 

y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor
bedID = bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed
table1ID = bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table
table2ID = bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table
# blockID = bc.loadURDF("cube.urdf", (-0.9, 0.2, 0), y2zOrn, useFixedBase=True, globalScaling=0.5)  # block on robot

# robot = UR5Robotiq85(bc, (-0.9, 0.15, 0.7), (-1.57, 0, 0))
# robot.load()
# robot.reset()


# position_control_group = []
# position_control_group.append(p.addUserDebugParameter('x', -1.0, 1.0, -0.807))
# position_control_group.append(p.addUserDebugParameter('y', 0.5, 1.5, 0.878))
# position_control_group.append(p.addUserDebugParameter('z', -0.5, 1.0, 0.56))
# position_control_group.append(p.addUserDebugParameter('roll', -3.14, 3.14, -1.57))
# position_control_group.append(p.addUserDebugParameter('pitch', -3.14, 3.14, 0))
# position_control_group.append(p.addUserDebugParameter('yaw', -3.14, 3.14, 3.14))
# position_control_group.append(p.addUserDebugParameter('gripper_opening', 0, 0.085, 0.08))


motion = MotionCaptureData()
motion.Load(motionPath)

humanoid = Humanoid(bc, motion, [0, 0.3, 0])
# human_motion_from_frame_data(humanoid, 60, bc)
# human_motion_from_frame_data_without_applypose(humanoid, 60, bc)
# time.sleep(1)
# print('getJointStateMultiDof: ', bc.getJointStateMultiDof(humanoid._humanoid, 3))
# print('getJointStateMultiDof: ', bc.getJointStateMultiDof(humanoid._humanoid, 4))


# pos, human_orn = bc.getLinkState(humanoid._humanoid, 4)[:2]  
# orn = bc.getQuaternionFromEuler([-1.57, 0, -1.57])   # NEED TO FIX THIS FOR DIFF INIT CONFIGS
# orn = bc.getEulerFromQuaternion(human_orn)
# print('eef orn: ', math.degrees(-1.57), math.degrees(0), math.degrees(-1.57))
# print('human orn: ', human_orn, " , ", math.degrees(orn[0]), math.degrees(orn[1]), math.degrees(orn[2]))
# print('orn: ', orn, " , ", math.degrees(bc.getEulerFromQuaternion(orn)))


while True:
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

  # parameter = []
  # for i in range(len(position_control_group)):
  #   parameter.append(bc.readUserDebugParameter(position_control_group[i]))

  # robot.move_ee(action=parameter[:-1], control_method='end')
  # robot.move_gripper(parameter[-1])

  bc.setJointMotorControlMultiDof(humanoid._humanoid,
                                  3,
                                  p.POSITION_CONTROL,
                                  # targetPosition=bc.getEulerFromQuaternion([-0.10111399739980698, 0.8756160140037537, -0.3728660047054291, 0.28990399837493896]))
                                  targetPosition=[-0.10111399739980698, 0.8756160140037537, -0.3728660047054291, 0.28990399837493896])
  bc.setJointMotorControlMultiDof(humanoid._humanoid,
                                  4,
                                  p.POSITION_CONTROL,
                                  targetPosition=[1.05714])

  p.stepSimulation()



### simulating human motion based on frame datasets
# simTime = 0
# keyFrameDuration = motion.KeyFrameDuraction()
# for utNum in range(motion.NumFrames()):
#   bc.stepSimulation()
#   humanoid.RenderReference(utNum * keyFrameDuration)  # RenderReference calls Slerp() & ApplyPose()
#   print('getJointStateMultiDof: ', bc.getJointStateMultiDof(humanoid._humanoid, 3))
#   time.sleep(0.01)



### range of feasible human joints
# rightElbows = [angle for angle in humanoid._rightElbowJointAnglesList]
# rightShoulders1 = [angle1 for angle1, angle2, angle3, angle4 in humanoid._rightShoulderJointAnglesList]
# rightShoulders2 = [angle2 for angle1, angle2, angle3, angle4 in humanoid._rightShoulderJointAnglesList]
# rightShoulders3 = [angle3 for angle1, angle2, angle3, angle4 in humanoid._rightShoulderJointAnglesList]
# rightShoulders4 = [angle4 for angle1, angle2, angle3, angle4 in humanoid._rightShoulderJointAnglesList]
# print('rightElbows min: ', min(rightElbows), 'max:', max(rightElbows))
# print('rightShoulders1 min: ', min(rightShoulders1), 'max:', max(rightShoulders1))
# print('rightShoulders2 min: ', min(rightShoulders2), 'max:', max(rightShoulders2))
# print('rightShoulders3 min: ', min(rightShoulders3), 'max:', max(rightShoulders3))
# print('rightShoulders4 min: ', min(rightShoulders4), 'max:', max(rightShoulders4))

Reset(humanoid)
p.disconnect()


