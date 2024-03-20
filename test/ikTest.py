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




# load environment
bc = BulletClient(connection_mode=p.GUI)

bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, -9.8, 0) 

y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor

# load humanoid
humanoid = bc.loadURDF("urdf/vhumanoid.urdf", [0, 0.9, 0], globalScaling=0.25, useFixedBase=True)

for i in range(bc.getNumJoints(humanoid)):
  print(i, bc.getLinkState(humanoid, i))
  print(i, bc.getJointStateMultiDof(humanoid, i))


# # shoulder (roll)
# target_pos = (0.0, 0.8424621943933351, -0.12762993742053488)
# target_orn = (0.5426857627753413, -0.0, -0.0, 0.8399358087859726)

# shoulder (roll) - elbow - wrist
target_pos = (7.450580596923828e-09, 0.6493250131607056, -0.11256325244903564)
target_orn =  (0.19753329455852509, -0.06968667358160019, 0.32530835270881653, 0.9221165180206299)


# test IK on humanoid
q_H = bc.calculateInverseKinematics(humanoid, 1, targetPosition=target_pos, targetOrientation=target_orn)
print('q_H ', q_H)

# move humanoid
# for i in range(bc.getNumJoints(humanoid)-1):
#   bc.setJointMotorControl2(humanoid, jointIndex=i, controlMode=p.POSITION_CONTROL, targetPosition=q_H[i])

bc.setJointMotorControlMultiDof(humanoid, jointIndex=0, controlMode=p.POSITION_CONTROL, targetPosition=bc.getQuaternionFromEuler(q_H[:3]))
bc.setJointMotorControl2(humanoid, jointIndex=1, controlMode=p.POSITION_CONTROL, targetPosition=q_H[3])






# load robot
robot = UR5Robotiq85(bc, (-1.0, 0.35, 0), (-1.57, 0, 0))
robot.load()
robot.reset()

# # test IK on robot
# target_pos =[ -0.18, 0.45, 0.2]
# target_orn = [0.592098320544675, 0.4078935042252701, 0.5958752005850314, 0.35773623432181634]

# q_R = bc.calculateInverseKinematics(robot.id, robot.eef_id, targetPosition=target_pos, targetOrientation=target_orn)
# print('q_R ', q_R)

# q_R = bc.calculateInverseKinematics(robot.id, robot.eef_id, target_pos, target_orn,
#                                                        robot.arm_lower_limits, robot.arm_upper_limits, robot.arm_joint_ranges, robot.arm_rest_poses,
#                                                        maxNumIterations=20)
# print('q_R ', q_R)
  

# test with simulation
time.sleep(1)

while (True):
  bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

  # print(bc.getLinkState(humanoid, 1)[4:6])

  bc.stepSimulation()
