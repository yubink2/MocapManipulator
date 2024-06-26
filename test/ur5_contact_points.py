import os, inspect
import pybullet as p
import sys
import numpy as np
import json
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
from utils.transform_utils import *

# UR5 parameters
LINK_SKELETON_NAME = [
    'shoulder_link',
    'upper_arm_link',
    'forearm_link',
    'wrist_1_link',
    'wrist_2_link',
    'wrist_3_link',
    'ee_link',
]

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
    T = compute_matrix(translation=point, rotation=[0,0,0,1])
    matrices.append(T)
  return matrices

# load environment
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.setGravity(0, 0, -9.8) 
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0))

# load robot
robot = UR5Robotiq85(bc, (0, 0, 0), (0, 0, 0))
robot.load()

# Control points for each link
shoulder_control_points = [[0,0,0], [0,0.055,0], [0,-0.055,0], [-0.055,0,0]]
upper_arm_control_points = [[0,0,0], #[0.12,0,0],
                            [0,0.055,0], [0,-0.055,0]
                            # [0.12,0.055,0], [0.12,-0.055,0], [-0.12,0.055,0], [-0.12,-0.055,0],
                            # [0.12,0,0.055], [0.12,0,-0.055], [-0.12,0,0.055], [-0.12,0,-0.055]
                            ]
forearm_control_points = [[0,0,0], #[-0.16,0,0], [-0.24,0,0],
                          [-0.05,-0.04,0], [-0.05,0.04,0], [0.05,-0.04,0], [0.05,0.04,0], 
                          # [-0.16,-0.04,0], [-0.16,0.04,0], [0.16, -0.04, 0], [0.16, 0.04, 0],
                          # [-0.24,0,-0.04], [-0.24,0,0.04], [-0.1,0,-0.04], [-0.1,0,0.04], [0.1,0,-0.04], [0.1,0,0.04]
                          ]
wrist_1_control_points = [[0,0,0], [0, 0, 0.03], [0, 0, -0.055], [0.03, 0, 0]]
wrist_2_control_points = [[0,0,0], [0.03, 0, 0], [-0.03, 0, 0], [0, 0, 0.055]]
wrist_3_control_points = [[0,0,0], [-0.03,0,0], [0.03, 0, 0], [0, 0, -0.03],
                          [-0.03,-0.05,0], [0.03, -0.05, 0], [0, -0.05, -0.03]]
ee_control_points = [[0,0,0], [0,0.12,0],
                     [0.03,0,0], [-0.03,0,0], [0, 0, 0.03], [0, 0, -0.03],
                     [0.055,0.055,0], [-0.055,0.055,0], [0.055,0.12,0], [-0.055,0.12,0]]

print(len(shoulder_control_points)+len(upper_arm_control_points)+len(forearm_control_points)+len(wrist_1_control_points)+len(wrist_2_control_points)+len(wrist_3_control_points)+len(ee_control_points))

# get link pose
shoulder_link = bc.getLinkState(robot.id, 1)[:2]
upper_arm_link = bc.getLinkState(robot.id, 2)[:2]
forearm_link = bc.getLinkState(robot.id, 3)[:2]
wrist_1_link = bc.getLinkState(robot.id, 4)[:2]
wrist_2_link = bc.getLinkState(robot.id, 5)[:2]
wrist_3_link = bc.getLinkState(robot.id, 6)[:2]
ee_link = bc.getLinkState(robot.id, robot.eef_id)[:2]

### Step 1: compute T_link_to_link_cp matrices
T_shoulder_to_world = inverse_matrix(compute_matrix(shoulder_link[0], shoulder_link[1]))
T_upper_arm_to_world = inverse_matrix(compute_matrix(upper_arm_link[0], upper_arm_link[1]))
T_forearm_to_world = inverse_matrix(compute_matrix(forearm_link[0], forearm_link[1]))
T_wrist_1_to_world = inverse_matrix(compute_matrix(wrist_1_link[0], wrist_1_link[1]))
T_wrist_2_to_world = inverse_matrix(compute_matrix(wrist_2_link[0], wrist_2_link[1]))
T_wrist_3_to_world = inverse_matrix(compute_matrix(wrist_3_link[0], wrist_3_link[1]))
T_ee_to_world = inverse_matrix(compute_matrix(ee_link[0], ee_link[1]))

T_world_to_shoulder_cp = []
for control_point in shoulder_control_points:
  pos = [shoulder_link[0][0]+control_point[0], shoulder_link[0][1]+control_point[1], shoulder_link[0][2]+control_point[2]]
  # draw_sphere_marker(bc, pos, radius=0.02, color=[0, 1, 0, 1])
  T_world_to_shoulder_cp.append(compute_matrix(pos, rotation=[0,0,0,1]))
T_world_to_shoulder_cp = np.asarray(T_world_to_shoulder_cp)
T_shoulder_to_shoulder_cp = np.matmul(T_shoulder_to_world, T_world_to_shoulder_cp)

T_world_to_upper_arm_cp = []
for control_point in upper_arm_control_points:
  pos = [upper_arm_link[0][0]+control_point[0], upper_arm_link[0][1]+control_point[1], upper_arm_link[0][2]+control_point[2]]
  # draw_sphere_marker(bc, pos, radius=0.02, color=[0, 0, 1, 1])
  T_world_to_upper_arm_cp.append(compute_matrix(pos, rotation=[0,0,0,1]))
T_world_to_upper_arm_cp = np.asarray(T_world_to_upper_arm_cp)
T_upper_arm_to_upper_arm_cp = np.matmul(T_upper_arm_to_world, T_world_to_upper_arm_cp)

T_world_to_forearm_cp = []
for control_point in forearm_control_points:
  pos = [forearm_link[0][0]+control_point[0], forearm_link[0][1]+control_point[1], forearm_link[0][2]+control_point[2]]
  # draw_sphere_marker(bc, pos, radius=0.02, color=[1, 0, 0, 1])
  T_world_to_forearm_cp.append(compute_matrix(pos, rotation=[0,0,0,1]))
T_world_to_forearm_cp = np.asarray(T_world_to_forearm_cp)
T_forearm_to_forearm_cp = np.matmul(T_forearm_to_world, T_world_to_forearm_cp)

T_world_to_wrist_1_cp = []
for control_point in wrist_1_control_points:
  pos = [wrist_1_link[0][0]+control_point[0], wrist_1_link[0][1]+control_point[1], wrist_1_link[0][2]+control_point[2]]
  # draw_sphere_marker(bc, pos, radius=0.02, color=[0, 1, 0, 1])
  T_world_to_wrist_1_cp.append(compute_matrix(pos, rotation=[0,0,0,1]))
T_world_to_wrist_1_cp = np.asarray(T_world_to_wrist_1_cp)
T_wrist_1_to_wrist_1_cp = np.matmul(T_wrist_1_to_world, T_world_to_wrist_1_cp)

T_world_to_wrist_2_cp = []
for control_point in wrist_2_control_points:
  pos = [wrist_2_link[0][0]+control_point[0], wrist_2_link[0][1]+control_point[1], wrist_2_link[0][2]+control_point[2]]
  # draw_sphere_marker(bc, pos, radius=0.02, color=[0, 0, 1, 1])
  T_world_to_wrist_2_cp.append(compute_matrix(pos, rotation=[0,0,0,1]))
T_world_to_wrist_2_cp = np.asarray(T_world_to_wrist_2_cp)
T_wrist_2_to_wrist_2_cp = np.matmul(T_wrist_2_to_world, T_world_to_wrist_2_cp)

T_world_to_wrist_3_cp = []
for control_point in wrist_3_control_points:
  pos = [wrist_3_link[0][0]+control_point[0], wrist_3_link[0][1]+control_point[1], wrist_3_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.02, color=[1, 0, 0, 1])
  T_world_to_wrist_3_cp.append(compute_matrix(pos, rotation=[0,0,0,1]))
T_world_to_wrist_3_cp = np.asarray(T_world_to_wrist_3_cp)
T_wrist_3_to_wrist_3_cp = np.matmul(T_wrist_3_to_world, T_world_to_wrist_3_cp)

T_world_to_ee_cp = []
for control_point in ee_control_points:
  pos = [ee_link[0][0]+control_point[0], ee_link[0][1]+control_point[1], ee_link[0][2]+control_point[2]]
  draw_sphere_marker(bc, pos, radius=0.02, color=[0, 1, 0, 1])
  T_world_to_ee_cp.append(compute_matrix(pos, rotation=[0,0,0,1]))
T_world_to_ee_cp = np.asarray(T_world_to_ee_cp)
T_ee_to_ee_cp = np.matmul(T_ee_to_world, T_world_to_ee_cp)


# ### Step 2: verify T_link_to_link_cp matrices
# # move robot
# bc.resetBasePositionAndOrientation(robot.id, [0.5, 0.7, 0], [0,0,0,1])
# q = [-2.7438, -1.6697,  1.1908, -1.1250, -0.0483,  1.8688]
# move_robot(bc, robot, q)

# # get new link pose
# shoulder_link = bc.getLinkState(robot.id, 1)[:2]
# upper_arm_link = bc.getLinkState(robot.id, 2)[:2]
# forearm_link = bc.getLinkState(robot.id, 3)[:2]
# wrist_1_link = bc.getLinkState(robot.id, 4)[:2]
# wrist_2_link = bc.getLinkState(robot.id, 5)[:2]
# wrist_3_link = bc.getLinkState(robot.id, 6)[:2]
# ee_link = bc.getLinkState(robot.id, robot.eef_id)[:2]

# T_world_to_shoulder = compute_matrix(shoulder_link[0], shoulder_link[1])
# T_world_to_upper_arm = compute_matrix(upper_arm_link[0], upper_arm_link[1])
# T_world_to_forearm = compute_matrix(forearm_link[0], forearm_link[1])
# T_world_to_wrist_1 = compute_matrix(wrist_1_link[0], wrist_1_link[1])
# T_world_to_wrist_2 = compute_matrix(wrist_2_link[0], wrist_2_link[1])
# T_world_to_wrist_3 = compute_matrix(wrist_3_link[0], wrist_3_link[1])
# T_world_to_ee = compute_matrix(ee_link[0], ee_link[1])

# # get new cp poses in world coordinate for each link
# T_world_to_shoulder_cp = np.matmul(T_world_to_shoulder, T_shoulder_to_shoulder_cp)
# T_world_to_upper_arm_cp = np.matmul(T_world_to_upper_arm, T_upper_arm_to_upper_arm_cp)
# T_world_to_forearm_cp = np.matmul(T_world_to_forearm, T_forearm_to_forearm_cp)
# T_world_to_wrist_1_cp = np.matmul(T_world_to_wrist_1, T_wrist_1_to_wrist_1_cp)
# T_world_to_wrist_2_cp = np.matmul(T_world_to_wrist_2, T_wrist_2_to_wrist_2_cp)
# T_world_to_wrist_3_cp = np.matmul(T_world_to_wrist_3, T_wrist_3_to_wrist_3_cp)
# T_world_to_ee_cp = np.matmul(T_world_to_ee, T_ee_to_ee_cp)

# # visualize control points
# for T_control_point in T_world_to_shoulder_cp:
#   pos = T_control_point[:3, 3]
#   draw_sphere_marker(bc, pos, radius=0.02, color=[0, 1, 0, 1])
# for T_control_point in T_world_to_upper_arm_cp:
#   pos = T_control_point[:3, 3]
#   draw_sphere_marker(bc, pos, radius=0.02, color=[0, 0, 1, 1])
# for T_control_point in T_world_to_forearm_cp:
#   pos = T_control_point[:3, 3]
#   draw_sphere_marker(bc, pos, radius=0.02, color=[1, 0, 0, 1])
# for T_control_point in T_world_to_wrist_1_cp:
#   pos = T_control_point[:3, 3]
#   draw_sphere_marker(bc, pos, radius=0.02, color=[0, 1, 0, 1])
# for T_control_point in T_world_to_wrist_2_cp:
#   pos = T_control_point[:3, 3]
#   draw_sphere_marker(bc, pos, radius=0.02, color=[0, 0, 1, 1])
# for T_control_point in T_world_to_wrist_3_cp:
#   pos = T_control_point[:3, 3]
#   draw_sphere_marker(bc, pos, radius=0.02, color=[1, 0, 0, 1])
# for T_control_point in T_world_to_ee_cp:
#   pos = T_control_point[:3, 3]
#   draw_sphere_marker(bc, pos, radius=0.02, color=[0, 1, 0, 1])

# #### DEBUGGING
# T_base_to_forearm = [[ 0.4018,  0.2791, -0.8721, -0.1197],
#          [ 0.1168, -0.9602, -0.2535, -0.0516],
#          [-0.9082,  0.0000, -0.4185,  0.4940],
#          [ 0.0000,  0.0000,  0.0000,  1.0000]]
# T_world_to_base = compute_matrix(translation=(0.5, 0.7, 0), rotation=(0,0,0), rotation_type='euler')

### Step 3: Write as json file
# generate json file
link_control_point_dict = {
  'shoulder_link': [],
  'upper_arm_link': [],
  'forearm_link': [],
  'wrist_1_link': [],
  'wrist_2_link': [],
  'wrist_3_link': [],
  'ee_link': [],
}

link_control_point_dict['shoulder_link'] = T_shoulder_to_shoulder_cp
link_control_point_dict['upper_arm_link'] = T_upper_arm_to_upper_arm_cp
link_control_point_dict['forearm_link'] = T_forearm_to_forearm_cp
link_control_point_dict['wrist_1_link'] = T_wrist_1_to_wrist_1_cp
link_control_point_dict['wrist_2_link'] = T_wrist_2_to_wrist_2_cp
link_control_point_dict['wrist_3_link'] = T_wrist_3_to_wrist_3_cp
link_control_point_dict['ee_link'] = T_ee_to_ee_cp

# Convert numpy arrays to lists
for key in link_control_point_dict:
  link_control_point_dict[key] = link_control_point_dict[key].tolist()

# Save dictionary to JSON file
with open('resources/ur5_control_points/T_control_points.json', 'w') as json_file:
  json.dump(link_control_point_dict, json_file, indent=4)