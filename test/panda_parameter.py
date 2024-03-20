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
bc.setTimestep = 0.0005

y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))

planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor
bedID = bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed
table1ID = bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table
table2ID = bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table

# load robot
robotID = bc.loadURDF('resources/panda/panda.urdf', (-1.0, -0.04, 0), y2zOrn, useFixedBase=True)
robot_eef = 8
for i in range(bc.getNumJoints(robotID)):
  print(i, bc.getLinkState(robotID, i))
  print(i, bc.getJointState(robotID, i))

# load humanoid
motion = MotionCaptureData()
motion.Load(motionPath)
humanoid = Humanoid(bc, motion, [0, 0.3, 0])

# cp pose
cp_pose = bc.getLinkState(humanoid._humanoid, 4)[:2] 
print('cp_pose: ', cp_pose)

# debug parameters
position_control_group = []
position_control_group.append(p.addUserDebugParameter('j1', -2.0, 2.0, -1.18))
position_control_group.append(p.addUserDebugParameter('j2', -2.0, 2.0, 0.45))
position_control_group.append(p.addUserDebugParameter('j3', -1.0, 1.0, 0.5))
position_control_group.append(p.addUserDebugParameter('j4', -3.14, 3.14, -1.3))
position_control_group.append(p.addUserDebugParameter('j5', -3.14, 3.14, 0.1))
position_control_group.append(p.addUserDebugParameter('j6', -3.14, 3.14, 1.5))

# open gripper
open_length = 0.04
for i in [9, 10]:
  bc.setJointMotorControl2(robotID, i, p.POSITION_CONTROL, open_length, force=20)

# print robot info
for i in range(bc.getNumJoints(robotID)):
  print(i, bc.getJointInfo(robotID, i))
  print(i, bc.getLinkState(robotID, i))

# visualize control points 
link1_control_points = [[0,-0.1294,0]]
link2_control_points = [[0,0,0.1294], [0,-0.194,0.055], [0,-0.194,-0.055], [0.055,-0.194,0], [-0.055,-0.194,0]]
link3_control_points = [[-0.03158,0,-0.02223], [0.0825,0.11105,0]]
link4_control_points = [[0,0,0.11055],[-0.11493,0.02223,0],[-0.1375,0.124,0],[-0.0275,0.124,0],[-0.0825,0.124,0.0545],[-0.0825,0.124,-0.0555]]
link5_control_points = [[0,0.1223,0.03067],[0,0.11113,-0.06748],[0,0.08161,-0.19864],[0,-0.0172,-0.1767]]
link6_control_points = [[-0.00458,0.0489,0],[-0.04783,0,0],[0,-0.0479,0],[0.00525,0,0.05628],[0.08835,0.07817,0],[0.13164,0.0171,0]]
link7_control_points = [[0.08084,0.04026,0.0853],[0.04026,0.08084,0.0853],[-0.03111,0.03111,0.0853],[0.03111,-0.03111,0.0853],[-0.03111,-0.03111,0.0853]]
link8_control_points = [[0.07176,0.07064,0.05771],[-0.0695,-0.07064,0.06041]]

base_pose = bc.getBasePositionAndOrientation(robotID)
link1_pose = bc.getLinkState(robotID, 0)[:2] 
link2_pose = bc.getLinkState(robotID, 1)[:2]
link3_pose = bc.getLinkState(robotID, 2)[:2]
link4_pose = bc.getLinkState(robotID, 3)[:2]
link5_pose = bc.getLinkState(robotID, 4)[:2]
link6_pose = bc.getLinkState(robotID, 5)[:2]
link7_pose = bc.getLinkState(robotID, 6)[:2]
link8_pose = bc.getLinkState(robotID, 7)[:2]

# draw_sphere_marker(bc, base_pose[0], radius=0.07, color=[1, 0, 0, 1])
# draw_sphere_marker(bc, link1_pose[0], radius=0.07, color=[1, 0, 0, 1])
# draw_sphere_marker(bc, link2_pose[0], radius=0.07, color=[0, 1, 0, 1])
# draw_sphere_marker(bc, link3_pose[0], radius=0.07, color=[0, 0, 1, 1])
# draw_sphere_marker(bc, link4_pose[0], radius=0.07, color=[1, 0, 0, 1])
# draw_sphere_marker(bc, link5_pose[0], radius=0.07, color=[0, 1, 0, 1])
# draw_sphere_marker(bc, link6_pose[0], radius=0.07, color=[0, 1, 0, 1])
# draw_sphere_marker(bc, link7_pose[0], radius=0.07, color=[0, 0, 1, 1])
draw_sphere_marker(bc, link8_pose[0], radius=0.07, color=[0, 0, 1, 1])

print('base_pose', base_pose[0])
print('link1_pose', link1_pose[0])
print('link2_pose', link2_pose[0])
print('link3_pose', link3_pose[0])
print('link4_pose', link4_pose[0])
print('link5_pose', link5_pose[0])
print('link7_pose', link7_pose[0])

# for control_point in link1_control_points:
#   pos = [link1_pose[0][0]+control_point[0], link1_pose[0][1]+control_point[1], link1_pose[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[1, 0, 0, 1])

# for control_point in link2_control_points:
#   pos = [link2_pose[0][0]+control_point[0], link2_pose[0][1]+control_point[1], link2_pose[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])

# for control_point in link3_control_points:
#   pos = [link3_pose[0][0]+control_point[0], link3_pose[0][1]+control_point[1], link3_pose[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])

# for control_point in link4_control_points:
#   pos = [link4_pose[0][0]+control_point[0], link4_pose[0][1]+control_point[1], link4_pose[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])

# for control_point in link5_control_points:
#   pos = [link5_pose[0][0]+control_point[0], link5_pose[0][1]+control_point[1], link5_pose[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])

# for control_point in link6_control_points:
#   pos = [link6_pose[0][0]+control_point[0], link6_pose[0][1]+control_point[1], link6_pose[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])

# for control_point in link7_control_points:
#   pos = [link7_pose[0][0]+control_point[0], link7_pose[0][1]+control_point[1], link7_pose[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])

for control_point in link8_control_points:
  pos = [link8_pose[0][0]+control_point[0], link8_pose[0][1]+control_point[1], link8_pose[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])


while True:
  parameter = []
  # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

  # parameter = []
  # for i in range(len(position_control_group)):
  #   parameter.append(bc.readUserDebugParameter(position_control_group[i]))

  # # robot.move_ee(action=parameter[:-1], control_method='end')

  # for i, joint_angle in enumerate(parameter):
  #   bc.setJointMotorControl2(robotID, i, p.POSITION_CONTROL, joint_angle)

  # # print('8) ', bc.getLinkState(robotID, 8))
  # # print('11) ', bc.getLinkState(robotID, 11))

  # # vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
  # # marker_id = bc.createMultiBody(basePosition=joint_8[0], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

  # # vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1])
  # # marker_id = bc.createMultiBody(basePosition=joint_11[0], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

  # p.stepSimulation()


Reset(humanoid)
p.disconnect()

# joint_8 = ((-0.5191241781128206, 0.43087502760960317, 0.45666145107722755), 
#           (0.8357188275941493, -0.04986700979146259, 0.05108630064394155, 0.5444974861518055))
# joint_11 = ((-0.516015913751437, 0.3393561632477693, 0.4164789155850513), 
#             (0.04986700979029541, 0.835718827594219, -0.5444974861518769, 0.0510863006431811))  # eef = 11
# joint_11 = ((-0.5196823277268179, 0.3667220578207198, 0.41692813317841715), 
#             (-0.06208896414048752, 0.821676027208712, -0.5664264377725382, -0.012432113045745374))
