import os, inspect
import os.path as osp
import pybullet as p
import math
import time
import sys
import torch
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
from transformation import compose_matrix, inverse_matrix

# humanoid
from env.motion_capture_data import MotionCaptureData
from humanoid import Humanoid
from humanoid import HumanoidPose


def draw_sphere_marker(bc, position, radius, color):
  vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
  marker_id = bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

def move_robot(bc, robot, q_robot):
  for _ in range(500):
    for i, joint_id in enumerate(robot.arm_controllable_joints):
      bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, q_robot[i],
                                force=robot.joints[joint_id].maxForce, maxVelocity=robot.joints[joint_id].maxVelocity)
    bc.stepSimulation()

def create_transformation_matrix(control_points):
  matrices = []
  for point in control_points:
    T = torch.eye(4, device='cuda:0')
    T[0, 3] = point[0]
    T[1, 3] = point[1]
    T[2, 3] = point[2]
    matrices.append(T)
  return torch.stack(matrices).to(device="cpu")

# load environment
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
# bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
# bc.setGravity(0, -9.8, 0) 
bc.setGravity(0, 0, -9.8) 

# y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
# planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0))

# load robot
# robot = UR5Robotiq85(bc, (-1.0, 0.35, 0), (-1.57, 0, 0))
# robot = UR5Robotiq85(bc, (-0.75, 0, 0), (-1.57, 0, 0))
robot = UR5Robotiq85(bc, (0, 0, 0), (0, 0, 0))
robot.load()

# Control points for each link
shoulder_control_points = [[0,0.055,0], [0,-0.055,0], [-0.055,0,0]]
upper_arm_control_points = [[0,0.055,0], [0,-0.055,0], [0.12,0.055,0], [0.12,-0.055,0], [-0.12,0.055,0], [-0.12,-0.055,0]]
forearm_control_points = [[-0.05,-0.06,0], [-0.05,0.06,0], [0.05,-0.06,0], [0.05,0.06,0], 
                          [-0.16,-0.06,0], [-0.16,0.06,0], [0.16, -0.06, 0], [0.16, 0.06, 0],
                          [-0.24,0,-0.06], [-0.24,0,0.06], [-0.1,0,-0.06], [-0.1,0,0.06], [0.1,0,-0.06], [0.1,0,0.06]]
wrist_1_control_points = [[0, 0, 0.07], [0, 0, -0.07], [0.07, 0, 0]]
wrist_2_control_points = [[0.06, 0, 0], [-0.06, 0, 0], [0, 0, 0.07]]
wrist_3_control_points = [[-0.06,0,0], [0.06, 0, 0], [0, 0, -0.06]]
ee_control_points = [[0.06,0,0], [-0.06,0,0], [0, 0, 0.06], [0, 0, -0.06],
                     [0.06,0.06,0], [-0.06,0.06,0], [0.06,0.12,0], [-0.06,0.12,0]]

print(len(shoulder_control_points)+len(upper_arm_control_points)+len(forearm_control_points)+len(wrist_1_control_points)+len(wrist_2_control_points)+len(wrist_3_control_points)+len(ee_control_points))

# Compute control point transformation matrices for each link
cp_shoulder = create_transformation_matrix(shoulder_control_points)
cp_upper_arm = create_transformation_matrix(upper_arm_control_points)
cp_forearm = create_transformation_matrix(forearm_control_points)
cp_wrist_1 = create_transformation_matrix(wrist_1_control_points)
cp_wrist_2 = create_transformation_matrix(wrist_2_control_points)
cp_wrist_3 = create_transformation_matrix(wrist_3_control_points)
cp_ee = create_transformation_matrix(ee_control_points)

# move robot
q = [-0.3896, -0.8765,  0.9869, -1.6000, -1.1553,  0.3350]
move_robot(bc, robot, q)


# get new link pose
shoulder_link = bc.getLinkState(robot.id, 1)[:2]
upper_arm_link = bc.getLinkState(robot.id, 2)[:2]
forearm_link = bc.getLinkState(robot.id, 3)[:2]
wrist_1_link = bc.getLinkState(robot.id, 4)[:2]
wrist_2_link = bc.getLinkState(robot.id, 5)[:2]
wrist_3_link = bc.getLinkState(robot.id, 6)[:2]
ee_link = bc.getLinkState(robot.id, robot.eef_id)[:2]

shoulder_w = torch.from_numpy(compose_matrix(translate=shoulder_link[0], angles=shoulder_link[1])).type(torch.FloatTensor).to(device="cpu")
upper_arm_w = torch.from_numpy(compose_matrix(translate=upper_arm_link[0], angles=upper_arm_link[1])).type(torch.FloatTensor).to(device="cpu")
forearm_w = torch.from_numpy(compose_matrix(translate=forearm_link[0], angles=forearm_link[1])).type(torch.FloatTensor).to(device="cpu")
wrist1_w = torch.from_numpy(compose_matrix(translate=wrist_1_link[0], angles=wrist_1_link[1])).type(torch.FloatTensor).to(device="cpu")
wrist2_w = torch.from_numpy(compose_matrix(translate=wrist_2_link[0], angles=wrist_2_link[1])).type(torch.FloatTensor).to(device="cpu")
wrist3_w = torch.from_numpy(compose_matrix(translate=wrist_3_link[0], angles=wrist_3_link[1])).type(torch.FloatTensor).to(device="cpu")
ee_w = torch.from_numpy(compose_matrix(translate=shoulder_link[0], angles=shoulder_link[1])).type(torch.FloatTensor).to(device="cpu")

# visualize control point locations
# shoulder_control_points = torch.matmul(shoulder_w, cp_shoulder)
# print(shoulder_control_points)
# for control_point in shoulder_control_points:
#   pos = control_point[:, 3]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[1, 0, 0, 1])

# upper_arm_control_points = torch.matmul(cp_upper_arm, upper_arm_w)
# print(upper_arm_control_points)
# for control_point in upper_arm_control_points:
#   pos = control_point[:, 3]
#   draw_sphere_marker(bc, pos, radius=0.05, color=[0, 1, 0, 1])

# forearm_control_points = torch.matmul(forearm_w, cp_forearm)
# print(forearm_control_points)
# for control_point in forearm_control_points:
#   pos = control_point[:, 3]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])


for control_point in shoulder_control_points:
  pos = [shoulder_link[0][0]+control_point[0], shoulder_link[0][1]+control_point[1], shoulder_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])

# for control_point in upper_arm_control_points:
#   pos = [upper_arm_link[0][0]+control_point[0], upper_arm_link[0][1]+control_point[1], upper_arm_link[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])

# for control_point in forearm_control_points:
#   pos = [forearm_link[0][0]+control_point[0], forearm_link[0][1]+control_point[1], forearm_link[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[1, 0, 0, 1])

# for control_point in wrist_1_control_points:
#   pos = [wrist_1_link[0][0]+control_point[0], wrist_1_link[0][1]+control_point[1], wrist_1_link[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])

# for control_point in wrist_2_control_points:
#   pos = [wrist_2_link[0][0]+control_point[0], wrist_2_link[0][1]+control_point[1], wrist_2_link[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 0, 1, 1])

# for control_point in wrist_3_control_points:
#   pos = [wrist_3_link[0][0]+control_point[0], wrist_3_link[0][1]+control_point[1], wrist_3_link[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[1, 0, 0, 1])

# for control_point in ee_control_points:
#   pos = [ee_link[0][0]+control_point[0], ee_link[0][1]+control_point[1], ee_link[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])

while True:
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

  # p.stepSimulation()
