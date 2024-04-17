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

# from humanoid import Humanoid
# from humanoid import HumanoidPose

from humanoid_with_rev import Humanoid
from humanoid_with_rev import HumanoidPose

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
  humanoid.RenderReference(utNum * keyFrameDuration, bc_arg)  # RenderReference calls Slerp() & ApplyPose()

def human_motion_from_frame_data_without_applypose(humanoid, utNum, bc_arg):
  keyFrameDuration = motion.KeyFrameDuraction()
  bc_arg.stepSimulation()
  pose = humanoid.RenderReferenceWithoutApplyPose(utNum * keyFrameDuration)
  print('--human_zmotion_from_frame_data_without_applypose: ', pose._rightShoulderRot)
  print('--human_zmotion_from_frame_data_without_applypose: ', pose._rightElbowRot)

def draw_sphere_marker(bc, position, radius=0.07, color=[1, 0, 0, 1]):
  vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
  marker_id = bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)
  return marker_id

def remove_marker(bc, marker_id):
    bc.removeBody(marker_id)

# # default
# dataset_path = 'data/data_3d_h36m.npz'
# motionPath = 'data/Sitting1.json'
# json_path = 'data/Sitting1.json'
# subject = 'S11'
# action = 'Sitting1'
# fps = 24
# loop = 'wrap'

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


right_shoulder_y = 3
right_shoulder_p = 4
right_shoulder_r = 5
right_elbow = 7

motion = MotionCaptureData()
motion.Load(motionPath)

humanoid = Humanoid(bc, motion, [0, 0.3, 0])

right_shoulder_rpy = [1.0, 1.0, 1.0]

world_to_cp = ((-0.5470086409113522, 0.3882300086040547, 0.4471872168793343), 
               (0.8014364705032705, 0.4986393080874306, 0.24895397868887204, -0.21698004671473697))
world_to_eef = ((-0.5414309985407247, 0.5279839286314739, 0.4593252144653385), 
                (-0.47718940632453627, 0.521248437497874, -0.4664003768983218, 0.5320347971014351))
eef_to_world = bc.invertTransform(world_to_eef[0], world_to_eef[1])
cp_to_eef = bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
                                        world_to_cp[0], world_to_cp[1])


for i in range(bc.getNumJoints(humanoid._humanoid)):
  print(bc.getJointInfo(humanoid._humanoid, i))

## TODO get feasible cp poses from datasets + robot goal pose
kwonji = 0
marker = None
while True:
   human_motion_from_frame_data(humanoid, kwonji, bc)
   world_to_new_cp = bc.getLinkState(humanoid._humanoid, right_elbow)[:2]
   world_to_new_eef = bc.multiplyTransforms(world_to_new_cp[0], world_to_new_cp[1],
                                        cp_to_eef[0], cp_to_eef[1])
   if marker is not None:
    remove_marker(bc, marker)
   marker = draw_sphere_marker(bc, world_to_new_eef[0])
   bc.stepSimulation()
   time.sleep(0.5)
   kwonji += 10

## TODO simulating human motion based on frame datasets
simTime = 0
keyFrameDuration = motion.KeyFrameDuraction()
for utNum in range(motion.NumFrames()):
# for utNum in range(100):
  bc.stepSimulation()
  humanoid.RenderReference(utNum * keyFrameDuration, bc)  # RenderReference calls Slerp() & ApplyPose()
  time.sleep(0.01)




## TODO range of feasible human joints
rightElbows = [angle for angle in humanoid._rightElbowJointAnglesList]
print('rightElbows min: ', min(rightElbows), 'max:', max(rightElbows))

# order: [roll, pitch, yaw]
rightShoulders1 = [angle1 for angle1, angle2, angle3 in humanoid._rightShoulderJointAnglesList]
rightShoulders2 = [angle2 for angle1, angle2, angle3 in humanoid._rightShoulderJointAnglesList]
rightShoulders3 = [angle3 for angle1, angle2, angle3 in humanoid._rightShoulderJointAnglesList]
print('rightShoulders1 min: ', min(rightShoulders1), 'max:', max(rightShoulders1))
print('rightShoulders2 min: ', min(rightShoulders2), 'max:', max(rightShoulders2))
print('rightShoulders3 min: ', min(rightShoulders3), 'max:', max(rightShoulders3))

Reset(humanoid)
p.disconnect()
sys.exit()





# TODO Test inverse kinematics
q_H = []
q_H.append(bc.getJointState(humanoid._humanoid, right_shoulder_r)[0])
q_H.append(bc.getJointState(humanoid._humanoid, right_shoulder_p)[0])
q_H.append(bc.getJointState(humanoid._humanoid, right_shoulder_y)[0])
q_H.append(bc.getJointState(humanoid._humanoid, right_elbow)[0])
print('current q_H: ', q_H)

cp_pos, cp_orn = bc.getLinkState(humanoid._humanoid, right_elbow)[:2]

shoulder_min = [-2.583238756496965, -0.248997453133789, -3.1402077384521765]
shoulder_max = [-1.3229245882839409, 1.2392816988875348, 3.1415394736319917]
elbow_min = [0.401146]
elbow_max = [2.541304]

#lower limits for null space
ll = shoulder_min + elbow_min
ll = list(np.array(q_H) - 0.1)
#upper limits for null space
ul = shoulder_max + elbow_max
ul = list(np.array(q_H) + 0.1)
#joint ranges for null space
jr = list(np.array(ul) - np.array(ll))
#restposes for null space
rp = q_H

q_H = bc.calculateInverseKinematics(humanoid._humanoid, right_elbow, 
                                    targetPosition=cp_pos, targetOrientation=cp_orn,
                                    # lowerLimits=ll, upperLimits=ul,
                                    # jointRanges=jr, restPoses=rp,
                                    # maxNumIterations=100, residualThreshold=0.0001
                                    )
print('calculated q_H from IK: ', q_H)
time.sleep(2)

bc.resetJointState(humanoid._humanoid, right_shoulder_y, q_H[2])
bc.resetJointState(humanoid._humanoid, right_shoulder_p, q_H[1])
bc.resetJointState(humanoid._humanoid, right_shoulder_r, q_H[0])
bc.resetJointState(humanoid._humanoid, right_elbow, q_H[3])
bc.stepSimulation()
print('moved to IK q_H')

print(bc.getJointState(humanoid._humanoid, right_shoulder_r)[0])
print(bc.getJointState(humanoid._humanoid, right_shoulder_p)[0])
print(bc.getJointState(humanoid._humanoid, right_shoulder_y)[0])
print(bc.getJointState(humanoid._humanoid, right_elbow)[0])

time.sleep(4)
sys.exit()