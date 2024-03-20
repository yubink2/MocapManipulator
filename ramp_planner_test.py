
import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

import numpy as np
import time

from mppi_planning.trajectory_planning import TrajectoryPlanner
from trajectory_following.trajectory_following import TrajectoryFollower

from mppi_planning.mppi_human_clamping import MPPI_H_Clamp

print('import done')

# Robot parameters
JOINT_LIMITS = [
    np.array([-2.8973, -1.7628, -2.8973, -
                3.0718, -2.8973, -0.0175, -2.8973]),
    np.array([2.8973, 1.7628, 2.8973, -
                0.0698, 2.8973, 3.7525, 2.8973])
]

LINK_FIXED = 'panda_link0'
LINK_EE = 'panda_hand'

LINK_SKELETON = [
    'panda_link1',
    'panda_link3',
    'panda_link4',
    'panda_link5',
    'panda_link7',
    'panda_hand',
]

ROBOT_Q_INIT = np.array(
    [
        -1.22151887,
        -1.54163973,
        -0.3665906,
        -2.23575787,
        0.5335327,
        1.04913162,
        -0.14688508,
        0.0,
        0.0,
    ]
)

ROBOT_Q_HOME = np.array(
    [
        0.0,            #Joint #1
        -0.785398,      #Joint #2
        0.0,            #Joint #3
        -1.5708,        #Joint #4
        0.0,            #Joint #5
        0.785398,       #Joint #6
        0.785398,       #Joint #7
        0.0,            #Gripper joint #1
        0.0,            #Gripper joint #2
    ]
)

robot_urdf_location = 'resources/panda/panda.urdf'
scene_urdf_location = 'resources/environment/environment.urdf'

print('resources loaded')

# Set up pybullet simulation
bc = BulletClient(connection_mode=p.GUI)

bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, -9.8, 0) 
# bc.setGravity(0, 0, -9.8) 

# load environment
y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor
# planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0))  # ground floor
robotID = bc.loadURDF(robot_urdf_location, (0, 0.05, 0), y2zOrn, useFixedBase=True)
robot_eef = 8

print(bc.getNumJoints(robotID))
for i in range(bc.getNumJoints(robotID)):
    print(bc.getJointInfo(robotID, i))


# Instantiate mppi human clamping object
eef_to_cp = ((-0.19717514514923096, 1.1558830738067627, -0.07056552171707153), 
            (0.7920243740081787, 0.13211965560913086, -0.5288172364234924, 0.27494409680366516))
shoulder_min = bc.getEulerFromQuaternion([-0.397618, -0.885139, -0.759182, -0.014004])
shoulder_max = bc.getEulerFromQuaternion([0.654265, 0.959910, 0.718185, 0.643939])
elbow_min = [0.4]
elbow_max = [2.5]
human_arm_lower_limits = shoulder_min + shoulder_max
human_arm_upper_limits = elbow_min + elbow_max
mppi_H_clamp = MPPI_H_Clamp(eef_to_cp, human_arm_lower_limits, human_arm_upper_limits)

# Instantiate trajectory planner
trajectory_planner = TrajectoryPlanner(
    joint_limits=JOINT_LIMITS,
    robot_urdf_location=robot_urdf_location,
    scene_urdf_location=scene_urdf_location,
    link_fixed=LINK_FIXED,
    link_ee=LINK_EE,
    link_skeleton=LINK_SKELETON,
    mppi_H_clamp=mppi_H_clamp,
)

print('trajectory planner instantiated')

# Trajectory Follower initialization
trajectory_follower = TrajectoryFollower(
    joint_limits = JOINT_LIMITS,
    robot_urdf_location = robot_urdf_location,
    link_fixed = LINK_FIXED,
    link_ee = LINK_EE,
    link_skeleton = LINK_SKELETON,
)

print('trajectory follower instantiated')


# MPPI parameters
N_JOINTS = 7
mppi_control_limits = [
    -0.05 * np.ones(N_JOINTS),
    0.05 * np.ones(N_JOINTS)
]
mppi_nsamples = 500
mppi_covariance = 0.005
mppi_lambda = 1.0

# init and goal joint angles
current_joint_angles = ROBOT_Q_INIT[:7]
# target_joint_angles = ROBOT_Q_HOME[:7]
target_pose = ((0.06353482091840526, -0.0006085134903523418, 0.8493233297970012), 
                (0.9957012796058373, -0.01136064747978677, 0.07291754005209923, -0.05597258108117516))
target_pose = ((0.35, 0.55, 0.35), (0, 0, 0, 1))
target_joint_angles = bc.calculateInverseKinematics(robotID, robot_eef, target_pose[0], target_pose[1])[:7]
print('target_joint_angles', target_joint_angles)

# mark goal pose
vs_id = bc.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])
marker_id = bc.createMultiBody(basePosition=target_pose[0], baseCollisionShapeIndex=-1, baseVisualShapeIndex=vs_id)

# Instantiate MPPI object
trajectory_planner.instantiate_mppi_ja_to_ja(
    current_joint_angles,
    target_joint_angles,
    mppi_control_limits=mppi_control_limits,
    mppi_nsamples=mppi_nsamples,
    mppi_covariance=mppi_covariance,
    mppi_lambda=mppi_lambda,
)
print('MPPI instantiated')


trajectory = trajectory_planner.get_mppi_rollout(current_joint_angles)
print(trajectory)

# set robot to initial joint positions
for i in range(N_JOINTS):
    bc.setJointMotorControl2(robotID, i, bc.POSITION_CONTROL, ROBOT_Q_INIT[i])

time.sleep(1)

print('traj: ', trajectory)
for q in trajectory:
    for i in range(N_JOINTS):
        bc.setJointMotorControl2(robotID, i, bc.POSITION_CONTROL, q[i])
    bc.stepSimulation() 
    time.sleep(0.1)

print('8) ', bc.getLinkState(robotID, 8))
print('11) ', bc.getLinkState(robotID, 11))
    

print('done')