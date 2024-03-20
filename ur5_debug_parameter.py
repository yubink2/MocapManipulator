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
sys.path.append("/usr/lib/python3/dist-packages")
import ompl



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
  keyFrameDuration = motion.KeyFrameDuraction()
  bc_arg.stepSimulation()
  humanoid.RenderReference(utNum * keyFrameDuration)

def human_motion_from_frame_data_without_applypose(humanoid, utNum, bc_arg):
  keyFrameDuration = motion.KeyFrameDuraction()
  bc_arg.stepSimulation()
  pose = humanoid.RenderReferenceWithoutApplyPose(utNum * keyFrameDuration)

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


dataset_path = 'data/data_3d_h36m.npz'
motionPath = 'data/Sitting1.json'
json_path = 'data/Sitting1.json'
subject = 'S11'
action = 'Sitting1'
fps = 24
loop = 'wrap'

bc = BulletClient(connection_mode=p.GUI)

bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, -9.8, 0) 

y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor
bedID = bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed
table1ID = bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table
table2ID = bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table

motion = MotionCaptureData()
motion.Load(motionPath)

humanoid = Humanoid(bc, motion, [0, 0.3, 0])
# human_motion_from_frame_data(humanoid, 60, bc)


print('jointInfo ', bc.getJointInfo(humanoid._humanoid, 3))
print(bc.getJointInfo(humanoid._humanoid, 4))
print(bc.getJointInfo(humanoid._humanoid, 5))

print('jointState ', bc.getJointState(humanoid._humanoid, 3))
print(bc.getJointState(humanoid._humanoid, 4))
print(bc.getJointState(humanoid._humanoid, 5))

print('linkState ', bc.getLinkState(humanoid._humanoid, 3))
print(bc.getLinkState(humanoid._humanoid, 4))
print(bc.getLinkState(humanoid._humanoid, 5))


# elbow_to_world = bc.getLinkState(humanoid._humanoid, 4)[4:6]  
# elbow_center_to_world = bc.getLinkState(humanoid._humanoid, 4)[:2]   
# wrist_to_world = bc.getLinkState(humanoid._humanoid, 5)[4:6]   

# world_to_shoulder = bc.invertTransform(shoulder_to_world[0], shoulder_to_world[1])
# elbow_to_shoulder = bc.multiplyTransforms(elbow_to_world[0], elbow_to_world[1],
#                                     world_to_shoulder[0], world_to_shoulder[1])
# elbow_to_shoulder = bc.invertTransform(elbow_to_shoulder[0], elbow_to_shoulder[1])

shoulder_to_world = bc.getLinkState(humanoid._humanoid, 3)[4:6]
shoulder_to_world_inertial = bc.getLinkState(humanoid._humanoid, 3)[2:4] 

elbow_to_world = bc.getLinkState(humanoid._humanoid, 4)[4:6]
elbow_to_world_inertial = bc.getLinkState(humanoid._humanoid, 4)[2:4] 

elbow_to_shoulder = bc.getJointInfo(humanoid._humanoid, 4)[14:16]
wrist_to_elbow = bc.getJointInfo(humanoid._humanoid, 5)[14:16]

# print('shoulder_to_world', shoulder_to_world)
# print('shoulder_to_world_inertial', shoulder_to_world_inertial)
# print('shoulder_to_world', shoulder_to_world)
# print('shoulder_to_world', shoulder_to_world)

elbow_to_shoulder = bc.multiplyTransforms(shoulder_to_world_inertial[0], shoulder_to_world_inertial[1],
                                          elbow_to_shoulder[0], elbow_to_shoulder[1])
# wrist_to_elbow = bc.multiplyTransforms(elbow_to_world_inertial[0], elbow_to_world_inertial[1],
#                                           wrist_to_elbow[0], wrist_to_elbow[1])
print('elbow_to_shoulder', elbow_to_shoulder)
print('wrist_to_elbow', wrist_to_elbow)

sphereRadius = 0.06
sphere_base = bc.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius)
sphere = bc.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius)
sphere_col = bc.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
mass = 1

# basePosition = shoulder_to_world[0]
# baseOrientation = shoulder_to_world[1]
# link_Masses = [0]
# linkCollisionShapeIndices = [-1]
# linkVisualShapeIndices = [sphere]
# linkPositions = [elbow_to_shoulder[0]]
# linkOrientations = [elbow_to_shoulder[1]]
# linkInertialFramePositions = [[0, 0, 0]]
# linkInertialFrameOrientations = [[0, 0, 0, 1]]
# indices = [0]
# jointTypes = [p.JOINT_REVOLUTE]
# axis = [[1, 1, 1]]

basePosition = shoulder_to_world[0]
baseOrientation = shoulder_to_world[1]
link_Masses = [1, 1]
linkCollisionShapeIndices = [-1, -1]
linkVisualShapeIndices = [sphere, sphere]
linkPositions = [elbow_to_shoulder[0], wrist_to_elbow[0]]
linkOrientations = [elbow_to_shoulder[1], wrist_to_elbow[1]]
linkInertialFramePositions = [[0, 0, 0], [0, 0, 0]]
linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
indices = [0, 1]
jointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
axis = [[0, 0, 1], [0, 0, 1]]

# sphereUid = bc.createMultiBody(mass,
#                               -1,
#                               sphere_base,
#                               basePosition,
#                               baseOrientation)

# sphereUid = bc.createMultiBody(mass,
#                               -1,
#                               sphere,
#                               basePosition,
#                               baseOrientation,
#                               linkMasses=link_Masses,
#                               linkCollisionShapeIndices=linkCollisionShapeIndices,
#                               linkVisualShapeIndices=linkVisualShapeIndices,
#                               linkPositions=linkPositions,
#                               linkOrientations=linkOrientations,
#                               linkInertialFramePositions=linkInertialFramePositions,
#                               linkInertialFrameOrientations=linkInertialFrameOrientations,
#                               linkParentIndices=indices,
#                               linkJointTypes=jointTypes,
#                               linkJointAxis=axis)


# print('base: ', basePosition, baseOrientation)
# print('link: ', linkPositions)
# print('link: ', linkOrientations)

# for i in range(bc.getNumJoints(sphereUid)):
#   print('*', bc.getJointInfo(sphereUid, i)[14:16])

# shoulder_joint = bc.getJointStateMultiDof(humanoid._humanoid, 3)[0]
# shoulder_joint = bc.getEulerFromQuaternion(shoulder_joint)
# elbow_joint = bc.getJointState(humanoid._humanoid, 4)[0]
# print(shoulder_joint, elbow_joint)

# p.setJointMotorControlMultiDof(sphereUid, 0, p.POSITION_CONTROL, targetPosition=shoulder_joint)

# for i in range(bc.getNumJoints(sphereUid)):
#   print('**', bc.getJointInfo(sphereUid, i)[14:16])

desired_eef_to_world_pos = bc.getLinkState(humanoid._humanoid, 4)[0]
desired_eef_to_world_pos = [desired_eef_to_world_pos[0], desired_eef_to_world_pos[1]+0.13, desired_eef_to_world_pos[2]]
elbow_to_world = bc.getLinkState(humanoid._humanoid, 4)[4:6]
cp_to_elbow = bc.getJointInfo(humanoid._humanoid, 5)[14:16]
cp_to_world = bc.multiplyTransforms(cp_to_elbow[0], cp_to_elbow[1],
                                    elbow_to_world[0], elbow_to_world[1])
cp_to_world = bc.getLinkState(humanoid._humanoid, 4)[:2]

print('desired_eef_to_world_pos: ', desired_eef_to_world_pos)
print('cp_to_world: ', cp_to_world)

while True:
  bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

  # bc.addUserDebugLine(elbow_to_world[0], cp_to_world[0], [1, 0, 0], 5.0)
  bc.addUserDebugLine(cp_to_world[0], desired_eef_to_world_pos, [1, 0, 0], 5.0)
  
  # bc.setJointMotorControl2(sphereUid, 1, p.POSITION_CONTROL, targetPosition=0.5, force=1000,maxVelocity=3)

  bc.stepSimulation()


Reset(humanoid)
p.disconnect()

