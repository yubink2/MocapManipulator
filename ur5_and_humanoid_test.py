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


def Reset(humanoid, bc_arg):
  global simTime
  humanoid.Reset()
  simTime = 0
  humanoid.SetSimTime(simTime)
  pose = humanoid.InitializePoseFromMotionData()
  humanoid.ApplyPose(pose, True, True, humanoid._humanoid, bc_arg)

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
  # print('--human_zmotion_from_frame_data_without_applypose: ', pose._rightShoulderRot)


dataset_path = 'data/data_3d_h36m.npz'
motionPath = 'data/Greeting.json'
json_path = 'data/Greeting.json'
subject = 'S11'
action = 'Greeting'
fps = 24
loop = 'wrap'


# dataset = init_fb_h36m_dataset(dataset_path)
# ground_truth = pose3D_from_fb_h36m(dataset, subject=subject, action=action, shift=[1.0, 0.0, 0.0])
# rot_seq = coord_seq_to_rot_seq(coord_seq=ground_truth, frame_duration=1 / fps)
# rot_seq_to_deepmimic_json(rot_seq=rot_seq, loop=loop, json_path=json_path)


# bc: main simulation
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, -9.8, 0) 

# bc_second: simulation (w/o visualization) to move human model on the side
bc_second = BulletClient(connection_mode=p.DIRECT)
bc_second.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc_second.setAdditionalSearchPath(pybullet_data.getDataPath())


robot = UR5Robotiq85(bc, (-1.0, 0.35, 0), (-1.57, 0, 0))
# robot = UR5Robotiq85(bc, (-0.8, 0.1, 0), (-1.57, 0, 0))
robot.load()
robot.reset()

y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor
bedID = bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed
table1ID = bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.3), y2zOrn, globalScaling=0.6)  # table
table2ID = bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.3), y2zOrn, globalScaling=0.6)  # table
blockID = bc.loadURDF("cube.urdf", (-1.0, 0.15, 0), y2zOrn, useFixedBase=True, globalScaling=0.45)  # block on robot


motion = MotionCaptureData()
motion.Load(motionPath)

humanoid = Humanoid(bc, motion, [0, 0.3, 0])
humanoid_second = Humanoid(bc_second, motion, [0, 0.3, 0])

human_motion_from_frame_data(humanoid, 0, bc)
human_motion_from_frame_data(humanoid_second, 0, bc_second)


pos, human_orn = bc.getLinkState(humanoid._humanoid, 4)[:2]  
pos_up = (pos[0], pos[1]+0.13, pos[2])
pos_up_2 = (pos[0], pos[1]+0.4, pos[2])

state_durations = [1 for _ in range(10)]
control_dt = 0.005  
bc.setTimestep = control_dt
state_t = 0.
current_state = 0

while True:  
  state_t += control_dt
  bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

  if (current_state == 0):
    robot.move_ee(action=pos_up_2 + (-1.57, 0, -1.57), control_method='end')

  if (current_state == 1):
    robot.move_ee(action=pos_up + (-1.57, 0.15, -1.57), control_method='end')

  if (current_state == 2):
    robot.move_gripper(0.065)

  if (current_state == 3):
    body_pose = bc.getLinkState(robot.id, robot.eef_id)
    obj_pose = bc.getLinkState(humanoid._humanoid, 4)
    world_to_body = bc.invertTransform(body_pose[0], body_pose[1])
    obj_to_body = bc.multiplyTransforms(world_to_body[0],
                                        world_to_body[1],
                                        obj_pose[0], obj_pose[1])

    cid = bc.createConstraint(parentBodyUniqueId=robot.id,
                        parentLinkIndex=robot.eef_id,
                        childBodyUniqueId=humanoid._humanoid,
                        childLinkIndex=4,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0))

  if (current_state == 4):
    eef_pos, eef_orn = bc.getLinkState(robot.id, robot.eef_id)[:2]
    eef_pos = [eef_pos[0], eef_pos[1]-0.13, eef_pos[2]]
    human_motion_from_frame_data(humanoid_second, 30, bc_second)
    time.sleep(0.001)
    goal_pose = bc_second.getLinkState(humanoid_second._humanoid, 4)[:2]
    bc.addUserDebugLine(lineFromXYZ=eef_pos, lineToXYZ=goal_pose[0], lineColorRGB=(255, 0, 0), lineWidth=2, lifeTime=0)
    body_pose = bc.getLinkState(robot.id, robot.eef_id)
    # world_to_body = bc.invertTransform(body_pose[0], body_pose[1])
    world_to_body = bc.invertTransform(eef_pos, eef_orn)
    obj_to_body = bc.multiplyTransforms(world_to_body[0],
                                        world_to_body[1],
                                        goal_pose[0], goal_pose[1])

  if (current_state == 5):                                      
    robot.move_ee(action=goal_pose[0] + bc.getEulerFromQuaternion(obj_to_body[1]), control_method='end')

  if (current_state == 6):
    human_motion_from_frame_data(humanoid_second, 45, bc_second)
    time.sleep(0.001)
    goal_pose = bc_second.getLinkState(humanoid_second._humanoid, 4)[:2]
    eef_pose = bc.getLinkState(robot.id, robot.eef_id)[:2]
    bc.addUserDebugLine(lineFromXYZ=eef_pose[0], lineToXYZ=goal_pose[0], lineColorRGB=(0, 255, 0), lineWidth=2, lifeTime=0)
    body_pose = bc.getLinkState(robot.id, robot.eef_id)
    world_to_body = bc.invertTransform(body_pose[0], body_pose[1])
    obj_to_body = bc.multiplyTransforms(world_to_body[0],
                                        world_to_body[1],
                                        goal_pose[0], goal_pose[1])

  if (current_state == 7):
    # robot.move_ee(action=goal_pose[0] + bc.getEulerFromQuaternion(goal_pose[1]), control_method='end')
    robot.move_ee(action=goal_pose[0] + bc.getEulerFromQuaternion(obj_to_body[1]), control_method='end')

  if state_t > state_durations[current_state]:
    current_state += 1
    if current_state >= len(state_durations):
      break
    state_t = 0

  bc.stepSimulation()




Reset(humanoid, bc)
Reset(humanoid_second, bc_second)
bc.removeConstraint(cid)
bc.disconnect()
bc_second.disconnect()



# 3 - right shoulder (middle)
# 4 - right elbow (middle-end part)
# 5 - right wrist (end part)
