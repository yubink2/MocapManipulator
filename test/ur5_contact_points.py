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

def human_motion_from_frame_data(humanoid, utNum, bc_arg):
  keyFrameDuration = motion.KeyFrameDuraction()
  bc_arg.stepSimulation()
  humanoid.RenderReference(utNum * keyFrameDuration, bc_arg)  # RenderReference calls Slerp() & ApplyPose()

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

# print robot info
for joint in robot.arm_controllable_joints:
  print(joint, bc.getJointInfo(robot.id, joint))
  print(joint, bc.getLinkState(robot.id, joint))

# TODO visualize control points
shoulder_link = bc.getLinkState(robot.id, 1)[:2]
upper_arm_link = bc.getLinkState(robot.id, 2)[:2]
forearm_link = bc.getLinkState(robot.id, 3)[:2]
wrist_1_link = bc.getLinkState(robot.id, 4)[:2]
wrist_2_link = bc.getLinkState(robot.id, 5)[:2]
wrist_3_link = bc.getLinkState(robot.id, 6)[:2]

shoulder_control_points = [[0,0,0.055], [0,0,-0.055], [-0.055,0,0]]
upper_arm_control_points = [[0,0,0.055], [0,0,-0.055], [0.055,0,0], [-0.055,0,0]]
forearm_control_points = [[-0.012,0,-0.095], [0.012,0,-0.095]]
wrist_1_control_points = [[0,0,0.04055], [-0.045,0.02223,0], [-0.0275,0.055,0]]
wrist_2_control_points = [[0,0.058,0.02067], [0,-0.0172,-0.05], [0.04,-0.0172,0]]
wrist_3_control_points = [[-0.00458,0.0489,0], [-0.04783,0,0], [0,-0.025,0.05], [0.00525,0,0.05628], [0.04535,0.03517,0], [0.053164,0.0171,0]]

print(len(shoulder_control_points)+len(upper_arm_control_points)+len(forearm_control_points)+len(wrist_1_control_points)+len(wrist_2_control_points)+len(wrist_3_control_points))

draw_sphere_marker(bc, shoulder_link[0], radius=0.06, color=[1, 0, 0, 1])
draw_sphere_marker(bc, upper_arm_link[0], radius=0.06, color=[0, 1, 0, 1])
draw_sphere_marker(bc, forearm_link[0], radius=0.06, color=[0, 0, 1, 1])
draw_sphere_marker(bc, wrist_1_link[0], radius=0.06, color=[1, 0, 0, 1])
draw_sphere_marker(bc, wrist_2_link[0], radius=0.06, color=[0, 1, 0, 1])
draw_sphere_marker(bc, wrist_3_link[0], radius=0.06, color=[0, 0, 1, 1])

for control_point in shoulder_link:
  pos = [shoulder_link[0][0]+control_point[0], shoulder_link[0][1]+control_point[1], shoulder_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])

for control_point in upper_arm_link:
  pos = [upper_arm_link[0][0]+control_point[0], upper_arm_link[0][1]+control_point[1], upper_arm_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])

for control_point in forearm_control_points:
  pos = [forearm_link[0][0]+control_point[0], forearm_link[0][1]+control_point[1], forearm_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.03, color=[1, 0, 0, 1])

for control_point in wrist_1_control_points:
  pos = [wrist_1_link[0][0]+control_point[0], wrist_1_link[0][1]+control_point[1], wrist_1_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.03, color=[1, 0, 0, 1])

for control_point in wrist_2_control_points:
  pos = [wrist_2_link[0][0]+control_point[0], wrist_2_link[0][1]+control_point[1], wrist_2_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.03, color=[1, 0, 0, 1])

for control_point in wrist_3_control_points:
  pos = [wrist_3_link[0][0]+control_point[0], wrist_3_link[0][1]+control_point[1], wrist_3_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.03, color=[1, 0, 0, 1])

while True:
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

  robot.reset()

  p.stepSimulation()
