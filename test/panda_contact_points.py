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

import pybullet_data
from pybullet_utils.bullet_client import BulletClient
import time
import numpy as np


def draw_sphere_marker(bc, position, radius, color):
  vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
  marker_id = bc.createMultiBody(basePosition=position, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

# control points
link1_control_points = [
        [
            0,
            -0.1294,
            0
        ]
    ]
link3_control_points = [
        [
            -0.03158,
            0,
            -0.02223
        ],
        [
            0.0825,
            0.11105,
            0
        ]
    ]
link4_control_points = [
        [
            0,
            0,
            0.11055
        ],
        [
            -0.11493,
            0.02223,
            0
        ],
        [
            -0.1375,
            0.124,
            0
        ],
        [
            -0.0275,
            0.124,
            0
        ],
        [
            -0.0825,
            0.124,
            0.0545
        ],
        [
            -0.0825,
            0.124,
            -0.0555
        ]
    ]
link5_control_points = [
        [
            0,
            0.1223,
            0.03067
        ],
        [
            0,
            0.11113,
            -0.06748
        ],
        [
            0,
            0.08161,
            -0.19864
        ],
        [
            0,
            -0.0172,
            -0.1767
        ]
    ]
link7_control_points = [
        [
            0.08084,
            0.04026,
            0.0853
        ],
        [
            0.04026,
            0.08084,
            0.0853
        ],
        [
            -0.03111,
            0.03111,
            0.0853
        ],
        [
            0.03111,
            -0.03111,
            0.0853
        ],
        [
            -0.03111,
            -0.03111,
            0.0853
        ]
    ]

# load environment
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, -9.8, 0) 

# load environment
y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor

# load robot
robotID = bc.loadURDF('resources/panda/panda.urdf', (-0.75, 0, 0), y2zOrn, useFixedBase=True)
robot_eef = 8
arm_controllable_joints = [1, 3, 4, 5, 7, 8]

for joint in range(bc.getNumJoints(robotID)):
  print(bc.getJointInfo(robotID, joint))

# TODO visualize control points
panda_link1 = bc.getLinkState(robotID, 0)[:2]
panda_link3 = bc.getLinkState(robotID, 2)[:2]
panda_link4 = bc.getLinkState(robotID, 3)[:2]
panda_link5 = bc.getLinkState(robotID, 4)[:2]
panda_link7 = bc.getLinkState(robotID, 6)[:2]
panda_hand = bc.getLinkState(robotID, 7)[:2]

# panda_link7 = bc.getLinkState(robotID, 7)[:2]

# draw_sphere_marker(bc, panda_link1[0], radius=0.06, color=[1, 0, 0, 1])
# draw_sphere_marker(bc, panda_link3[0], radius=0.06, color=[0, 1, 0, 1])
# draw_sphere_marker(bc, panda_link4[0], radius=0.06, color=[0, 0, 1, 1])
# draw_sphere_marker(bc, panda_link5[0], radius=0.06, color=[1, 0, 0, 1])
# draw_sphere_marker(bc, panda_link7[0], radius=0.06, color=[0, 1, 0, 1])
# draw_sphere_marker(bc, panda_hand[0], radius=0.06, color=[0, 0, 1, 1])

for control_point in link1_control_points :
  pos = [panda_link1[0][0]+control_point[0], panda_link1[0][1]+control_point[1], panda_link1[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])
  draw_sphere_marker(bc, control_point, radius=0.03, color=[0, 1, 0, 1])

for control_point in link3_control_points :
  pos = [panda_link3[0][0]+control_point[0], panda_link3[0][1]+control_point[1], panda_link3[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])
  draw_sphere_marker(bc, control_point, radius=0.03, color=[0, 1, 0, 1])

for control_point in link4_control_points :
  pos = [panda_link4[0][0]+control_point[0], panda_link4[0][1]+control_point[1], panda_link4[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])
  draw_sphere_marker(bc, control_point, radius=0.03, color=[0, 1, 0, 1])

for control_point in link5_control_points :
  pos = [panda_link5[0][0]+control_point[0], panda_link5[0][1]+control_point[1], panda_link5[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])
  draw_sphere_marker(bc, control_point, radius=0.03, color=[0, 1, 0, 1])

for control_point in link7_control_points :
  pos = [panda_link7[0][0]+control_point[0], panda_link7[0][1]+control_point[1], panda_link7[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])
  draw_sphere_marker(bc, control_point, radius=0.03, color=[0, 1, 0, 1])

# for control_point in link7_control_points:
#   pos = [panda_link7[0][0]+control_point[0], panda_link7[0][1]+control_point[1], panda_link7[0][2]+control_point[2]]
#   draw_sphere_marker(bc, pos, radius=0.03, color=[0, 1, 0, 1])

  


while True:
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
  p.stepSimulation()