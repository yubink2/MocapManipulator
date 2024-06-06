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
from humanoid_with_rev import Humanoid
from humanoid_with_rev import HumanoidPose


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
bedID = bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.0)  # bed
table1ID = bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table
table2ID = bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.6), y2zOrn, globalScaling=0.6)  # table

# # set camera
bc.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=135.000, cameraPitch=-45.00, cameraTargetPosition=(-1.5, 1.8, 1.5))

# load robot
robot = UR5Robotiq85(bc, (-0.75, 0, 0), (-1.57, 0, 0), globalScaling=1.2)
robot.load()
robot.reset()

# load humanoid
motion = MotionCaptureData()
motion.Load(motionPath)
humanoid = Humanoid(bc, motion, [0, 0.3, 0])

right_shoulder_y = 3
right_shoulder_p = 4
right_shoulder_r = 5
right_elbow = 7

human_motion_from_frame_data(humanoid, 230, bc)


# # TODO move robot to grasp pose
# pos, orn = bc.getLinkState(humanoid._humanoid, right_elbow)[:2]
# pos_up = (pos[0], pos[1]+0.16, pos[2]-0.05)
# orn = bc.getQuaternionFromEuler((-1.57, 0.35, -1.75))

# # TODO move robot to grasp pose
# pos, orn = bc.getLinkState(humanoid._humanoid, right_elbow)[:2]
# pos_up = (pos[0], pos[1]+0.165, pos[2]-0.1)
# orn = bc.getQuaternionFromEuler((-1.2, -0.6, -1.8))

# # TODO move robot to grasp pose (60)
# pos, orn = bc.getLinkState(humanoid._humanoid, right_elbow)[:2]
# print(pos)
# pos_up_2 = (pos[0]-0.4, pos[1]+0.072, pos[2]+0.04)
# pos_up = (pos[0]-0.18, pos[1]+0.072, pos[2]+0.04)
# orn = bc.getQuaternionFromEuler((0.890, 0.210, -0.339))

# TODO move robot to grasp pose (230)
pos, orn = bc.getLinkState(humanoid._humanoid, right_elbow)[:2]
print('pos: ', pos)
pos_up_2 = (-0.17962980178173082, 0.7, 0.1)
pos_up = (-0.17962980178173082, 0.590458026205689, 0.3667859122611105)
orn = (0.42581023056381473, 0.025895246484703916, 0.8784134154854197, -0.21541809406808593)



## TODO move robot to pos_up_2
current_joint_angles = bc.calculateInverseKinematics(robot.id, robot.eef_id, pos_up_2, orn,
                                                robot.arm_lower_limits, robot.arm_upper_limits, robot.arm_joint_ranges, robot.arm_rest_poses,
                                                maxNumIterations=20)
current_joint_angles = [current_joint_angles[i] for i in range(len(robot.arm_controllable_joints))]

for _ in range (100):
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, current_joint_angles[i],
                                        force=robot.joints[joint_id].maxForce, maxVelocity=robot.joints[joint_id].maxVelocity)
    bc.stepSimulation()

### move robot to pos_up
current_joint_angles = bc.calculateInverseKinematics(robot.id, robot.eef_id, pos_up, orn,
                                                robot.arm_lower_limits, robot.arm_upper_limits, robot.arm_joint_ranges, robot.arm_rest_poses,
                                                maxNumIterations=20)
current_joint_angles = [current_joint_angles[i] for i in range(len(robot.arm_controllable_joints))]

for _ in range (100):
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        bc.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, current_joint_angles[i],
                                        force=robot.joints[joint_id].maxForce, maxVelocity=robot.joints[joint_id].maxVelocity)
    bc.stepSimulation()
print('moved robot to init config')

# attach human arm (obj) to eef (body)
body_pose = bc.getLinkState(robot.id, robot.eef_id)  # world to eef 
obj_pose = bc.getLinkState(humanoid._humanoid, right_elbow)  # world to cp
world_to_body = bc.invertTransform(body_pose[0], body_pose[1])  # eef to world
obj_to_body = bc.multiplyTransforms(world_to_body[0],          # eef to cp
                                    world_to_body[1],
                                    obj_pose[0], obj_pose[1])

cid = bc.createConstraint(parentBodyUniqueId=robot.id,
                    parentLinkIndex=robot.eef_id,
                    childBodyUniqueId=humanoid._humanoid,
                    childLinkIndex=right_elbow,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_to_body[0],
                    parentFrameOrientation=obj_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0))


# TODO manually move robot and print its joint configs
while True:
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

  for i, joint_id in enumerate(robot.arm_controllable_joints):
    print(i, bc.getJointState(robot.id, joint_id)[0])

  p.stepSimulation()





# TODO move to target joint
# target_pos = (-0.505, 0.268, -0.395)
# target_orn = bc.getQuaternionFromEuler((-0.297, -0.826, -1.686))
# current_joint_angles = bc.calculateInverseKinematics(robot.id, robot.eef_id, target_pos, target_orn,
#                                                 robot.arm_lower_limits, robot.arm_upper_limits, robot.arm_joint_ranges, robot.arm_rest_poses,
#                                                 maxNumIterations=20)
# current_joint_angles = [current_joint_angles[i] for i in range(len(robot.arm_controllable_joints))]
# print(current_joint_angles)

