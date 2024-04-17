import os, inspect
import os.path as osp
import pybullet as p
import math
import numpy as np
import time
import sys
sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

# ur5
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
# bedID = bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed
# table1ID = bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table
# table2ID = bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table

motion = MotionCaptureData()
motion.Load(motionPath)


########################################################################################

# # TODO test humanoid joint rotation

# humanoid_sph = bc.loadURDF("humanoid/humanoid.urdf", [0, 0.9, 0], 
#                         globalScaling=0.25,
#                         useFixedBase=True)

# humanoid_rev = bc.loadURDF("urdf/humanoid_with_rev.urdf", [-1.5, 0.9, 0],
#                         globalScaling=0.25,
#                         useFixedBase=True)

# rightShoulder = 3
# rightElbow = 4

# right_shoulder_r = 3
# right_shoulder_p = 4
# right_shoulder_y = 5
# right_elbow = 7

# # q_H_test = [0, 1, 0, 0, 0.45577464302391746]
# q_H_test = [-1.8130499045101367, -0.01646934556678218, 2.484668672024347, 0.45]
# q_H_test = [1.0, 1.0, 1.0, 0.45]

# for i in range(bc.getNumJoints(humanoid_rev)):
#   print(bc.getJointInfo(humanoid_rev, i))

# while(True):
#   # # move humanoid_spherical
#   # bc.setJointMotorControlMultiDof(humanoid_sph, rightShoulder, controlMode=p.POSITION_CONTROL, targetPosition=bc.getQuaternionFromEuler(q_H_test[:3]))
#   # bc.setJointMotorControl2(humanoid_sph, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[3])

#   # # move humanoid_revolute
#   # bc.setJointMotorControl2(humanoid_rev, right_shoulder_r, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[0])
#   # bc.setJointMotorControl2(humanoid_rev, right_shoulder_p, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[1])
#   # bc.setJointMotorControl2(humanoid_rev, right_shoulder_y, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[2])
#   # bc.setJointMotorControl2(humanoid_rev, right_elbow, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[3])

#   time.sleep(0.1)
#   bc.stepSimulation()

# sys.exit()

# humanoid = bc.loadURDF("urdf/humanoid_with_rev.urdf", [0, 0.9, 0],
#                         globalScaling=0.25,
#                         useFixedBase=True)

# right_shoulder_r = 3
# right_shoulder_p = 4
# right_shoulder_y = 5
# rightElbow = 7
# q_H_test = [3.14, 0, 3.14, 0.45577464302391746]
# q_H_test = [-1.8130499045101367, -0.01646934556678218, 2.484668672024347, 0.45]
# # q_H_test = [1.8130499045101367, -0.01646934556678218, -2.484668672024347, 0.45]

# while(True):
#   bc.setJointMotorControl2(humanoid, right_shoulder_r, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[0])
#   bc.setJointMotorControl2(humanoid, right_shoulder_p, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[1])
#   bc.setJointMotorControl2(humanoid, right_shoulder_y, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[2])
#   bc.setJointMotorControl2(humanoid, rightElbow, controlMode=p.POSITION_CONTROL, targetPosition=q_H_test[3])
#   bc.stepSimulation()

########################################################################################

# # TODO frame transformation & use visual debugger

# pos, orn = bc.getLinkState(humanoid._humanoid, 4)[:2]  
# pos_up = [pos[0], pos[1]+0.13, pos[2]]
# orn = bc.getQuaternionFromEuler([-1.57, 0.15, -1.57])

# # eef_to_world = bc.getLinkState(robot.id, robot.eef_id)
# cubeId = p.loadURDF("cube_small.urdf", pos_up, orn)
# # print('cubeId: ', bc.getBasePositionAndOrientation(cubeId))

# body_pose = bc.getBasePositionAndOrientation(cubeId) 
# obj_pose = bc.getLinkState(humanoid._humanoid, 4)  
# world_to_body = bc.invertTransform(body_pose[0], body_pose[1])  
# obj_to_body = bc.multiplyTransforms(world_to_body[0],
#                                     world_to_body[1],
#                                     obj_pose[0], obj_pose[1]) 

# cid = bc.createConstraint(parentBodyUniqueId=humanoid._humanoid,
#                     parentLinkIndex=4,
#                     childBodyUniqueId=cubeId,
#                     childLinkIndex=-1,
#                     jointType=p.JOINT_FIXED,
#                     jointAxis=(0, 0, 0),
#                     parentFramePosition=obj_to_body[0],
#                     parentFrameOrientation=obj_to_body[1],
#                     childFramePosition=(0, 0, 0),
#                     childFrameOrientation=(0, 0, 0))

# eef_to_world = bc.getBasePositionAndOrientation(cubeId)
# cp_to_world = bc.getLinkState(humanoid._humanoid, 4)
# world_to_cp = bc.invertTransform(cp_to_world[0], cp_to_world[1])
# eef_to_cp = bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
#                                   world_to_cp[0], world_to_cp[1])

# eef_to_world = bc.multiplyTransforms(eef_to_cp[0], eef_to_cp[1],
#                                     cp_to_world[0], cp_to_world[1])


# print('correct pose: ', bc.getBasePositionAndOrientation(cubeId))
# print(eef_to_world)

# time.sleep(1)
# human_motion_from_frame_data(humanoid, 60, bc)

# cp_to_world = bc.getLinkState(humanoid._humanoid, 4)
# eef_to_world = bc.multiplyTransforms(eef_to_cp[0], eef_to_cp[1],
#                                     cp_to_world[0], cp_to_world[1])

# ########################################################################################

# # TODO inverse kinematics on humanoid (revolute)

# humanoid = bc.loadURDF("urdf/humanoid_simplified_rev.urdf", [0, 0.9, 0],
#                         globalScaling=0.25,
#                         useFixedBase=True)

# for i in range(bc.getNumJoints(humanoid)):
#   print(bc.getJointInfo(humanoid, i))

# right_shoulder_r = 0
# right_shoulder_p = 1
# right_shoulder_y = 2
# rightElbow = 4

# init_q_H = [-1.8130497863744381, -0.01646932817665812, 2.4846686720243474, 0.45445953888343765]
# for _ in range(100):
#    bc.setJointMotorControl2(humanoid, right_shoulder_r, p.POSITION_CONTROL, init_q_H[0])
#    bc.setJointMotorControl2(humanoid, right_shoulder_p, p.POSITION_CONTROL, init_q_H[1])
#    bc.setJointMotorControl2(humanoid, right_shoulder_y, p.POSITION_CONTROL, init_q_H[2])
#    bc.setJointMotorControl2(humanoid, rightElbow, p.POSITION_CONTROL, init_q_H[3])
#    bc.stepSimulation()
# print('moved to init q_H')

# print(bc.getJointState(humanoid, right_shoulder_r)[0])
# print(bc.getJointState(humanoid, right_shoulder_p)[0])
# print(bc.getJointState(humanoid, right_shoulder_y)[0])
# print(bc.getJointState(humanoid, rightElbow)[0])

# q_H = []
# q_H.append(bc.getJointState(humanoid, right_shoulder_r)[0])
# q_H.append(bc.getJointState(humanoid, right_shoulder_p)[0])
# q_H.append(bc.getJointState(humanoid, right_shoulder_y)[0])
# q_H.append(bc.getJointState(humanoid, rightElbow)[0])
# print(q_H)

# cp_pos, cp_orn = bc.getLinkState(humanoid, rightElbow)[:2]

# shoulder_min = [-2.583238756496965, -0.248997453133789, -3.1402077384521765]
# shoulder_max = [-1.3229245882839409, 1.2392816988875348, 3.1415394736319917]
# elbow_min = [0.401146]
# elbow_max = [2.541304]

# #lower limits for null space
# ll = shoulder_min + elbow_min
# #upper limits for null space
# ul = shoulder_max + elbow_max
# #joint ranges for null space
# jr = list(np.array(ul) - np.array(ll))
# #restposes for null space
# rp = q_H

# q_H = bc.calculateInverseKinematics(humanoid, rightElbow, 
#                                     targetPosition=cp_pos, targetOrientation=cp_orn,
#                                     lowerLimits=ll, upperLimits=ul,
#                                     jointRanges=jr, restPoses=rp,
#                                     maxNumIterations=100, residualThreshold=0.0001
#                                     )
# print('calculated q_H from IK: ', q_H)
# time.sleep(4)

# for _ in range(100):
#    bc.setJointMotorControl2(humanoid, right_shoulder_r, p.POSITION_CONTROL, q_H[0])
#    bc.setJointMotorControl2(humanoid, right_shoulder_p, p.POSITION_CONTROL, q_H[1])
#    bc.setJointMotorControl2(humanoid, right_shoulder_y, p.POSITION_CONTROL, q_H[2])
#    bc.setJointMotorControl2(humanoid, rightElbow, p.POSITION_CONTROL, q_H[3])
#    bc.stepSimulation()
# print('moved to IK q_H')

# print(bc.getJointState(humanoid, right_shoulder_r)[0])
# print(bc.getJointState(humanoid, right_shoulder_p)[0])
# print(bc.getJointState(humanoid, right_shoulder_y)[0])
# print(bc.getJointState(humanoid, rightElbow)[0])

# time.sleep(4)
# sys.exit()

# ########################################################################################

# # TODO inverse kinematics on humanoid (spherical)

# humanoid = bc.loadURDF("urdf/humanoid.urdf", [0, 0.9, 0],
#                         globalScaling=0.25,
#                         useFixedBase=True)

humanoid = bc.loadURDF("urdf/humanoid_simplified_sph.urdf", [0, 0.9, 0],
                        globalScaling=0.25,
                        useFixedBase=True)

for i in range(bc.getNumJoints(humanoid)):
  print(bc.getJointInfo(humanoid, i))

# rightShoulder = 3
# rightElbow = 4
  
rightShoulder = 0
rightElbow = 1

init_q_H = [-1.8130497863744381, -0.01646932817665812, 2.4846686720243474, 0.45445953888343765]
for _ in range(500):
   bc.setJointMotorControlMultiDof(humanoid, rightShoulder, p.POSITION_CONTROL, bc.getQuaternionFromEuler(init_q_H[:3]))
   bc.setJointMotorControl2(humanoid, rightElbow, p.POSITION_CONTROL, init_q_H[3])
   bc.stepSimulation()
print('moved to init q_H')

print(bc.getEulerFromQuaternion(bc.getJointStateMultiDof(humanoid, rightShoulder)[0]))
print(bc.getJointState(humanoid, rightElbow)[0])

q_H = list(bc.getEulerFromQuaternion(bc.getJointStateMultiDof(humanoid, rightShoulder)[0])) + [bc.getJointState(humanoid, rightElbow)[0]]
print('current q_H: ', q_H)

cp_pos, cp_orn = bc.getLinkState(humanoid, rightElbow)[:2]

shoulder_min = [-2.583238756496965, -0.248997453133789, -3.1402077384521765]
shoulder_max = [-1.3229245882839409, 1.2392816988875348, 3.1415394736319917]
elbow_min = [0.401146]
elbow_max = [2.541304]

#lower limits for null space
ll = shoulder_min + elbow_min
# ll = list(np.array(q_H) - 0.1)
#upper limits for null space
ul = shoulder_max + elbow_max
# ul = list(np.array(q_H) + 0.1)
#joint ranges for null space
jr = list(np.array(ul) - np.array(ll))
#restposes for null space
rp = q_H

q_H = bc.calculateInverseKinematics(humanoid, rightElbow, 
                                    targetPosition=cp_pos, targetOrientation=cp_orn,
                                    lowerLimits=ll, upperLimits=ul,
                                    jointRanges=jr, restPoses=rp,
                                    maxNumIterations=100, residualThreshold=0.0001
                                    )
print('calculated q_H from IK: ', q_H)
time.sleep(4)
sys.exit()

for _ in range(100):
   bc.setJointMotorControl2(humanoid, right_shoulder_r, p.POSITION_CONTROL, q_H[0])
   bc.setJointMotorControl2(humanoid, right_shoulder_p, p.POSITION_CONTROL, q_H[1])
   bc.setJointMotorControl2(humanoid, right_shoulder_y, p.POSITION_CONTROL, q_H[2])
   bc.setJointMotorControl2(humanoid, rightElbow, p.POSITION_CONTROL, q_H[3])
   bc.stepSimulation()
print('moved to IK q_H')

print(bc.getJointState(humanoid, right_shoulder_r)[0])
print(bc.getJointState(humanoid, right_shoulder_p)[0])
print(bc.getJointState(humanoid, right_shoulder_y)[0])
print(bc.getJointState(humanoid, rightElbow)[0])

time.sleep(4)
sys.exit()

# ########################################################################################

# while True:
#   bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

#   # after = bc.getLinkState(humanoid._humanoid, rightElbow)[:2]
#   # print('target: ', target)
#   # print('after: ', after)

#   bc.stepSimulation()


# ## simulating human motion based on frame datasets
# simTime = 0
# keyFrameDuration = motion.KeyFrameDuraction()
# for utNum in range(motion.NumFrames()):
#   bc.stepSimulation()
#   humanoid.RenderReference(utNum * keyFrameDuration) 
#   time.sleep(0.01)


# Reset(humanoid)
p.disconnect()

