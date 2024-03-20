import pybullet as p
import time
import numpy as np
from pybullet_utils.bullet_client import BulletClient
import pybullet_data
    

bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, 0, 0) 
bc.setTimestep = 0.0005
y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
planeID = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)  # ground floor

# sphereRadius = 0.06
# base_sphere_vis = bc.createVisualShape(p.GEOM_SPHERE, radius=0.08)
# base_sphere_col = bc.createCollisionShape(p.GEOM_SPHERE, radius=0.08)
# sphere_vis = -1
# sphere_col = bc.createCollisionShape(p.GEOM_BOX,halfExtents=[0.06, 0.06, 0.06])
box_vis = -1
box_col = bc.createCollisionShape(p.GEOM_BOX,halfExtents=[0.05, 0.05, 0.05])

mass = 1

joint_col = -1
joint_vis = -1
joint_pos = [0, 0, 0]
joint_orn = [0, 0, 0, 1]

# basePosition = [-0.16845570504665375, 0.3219473361968994, 0.48509156703948975]
# baseOrientation = [0.6721086502075195, 0.6636524796485901, 0.28768390417099, -0.15834574401378632]
# linkMasses = [1, 1, 1, 1, 1]
# linkCollisionShapeIndices = [box_vis, box_vis, box_col, box_col, box_col]
# linkVisualShapeIndices = [box_vis, box_vis, box_vis, box_vis, box_vis]
# linkPositions = [[0, 0, 0], [0, 0, 0], (0.0, -0.2747880220413208, 0.0), (0.0, -0.138947, 0.0), (2.980232238769531e-07, -2.384185791015625e-07, 0.1299997866153717)]
# linkOrientations = [[0, 0, 0, 1], [0, 0, 0, 1], (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, -2.60770320892334e-08, 5.587935447692871e-09, 1.0)]
# linkInertialFramePositions = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
# linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
# linkParentIndices = [0, 1, 2, 3, 4]
# linkJointTypes = [0, 0, 0, 0, 0]
# linkJointAxis = [[1, 0, 0], [0, 1, 0], [0, 0, 1], (0.0, 0.0, 1.0), [0, 1, 0]]

basePosition = [-0.16845570504665375, 0.3219473361968994, 0.48509156703948975]
baseOrientation = [0.6721086502075195, 0.6636524796485901, 0.28768390417099, -0.15834574401378632]
linkMasses = [1, 1, 1, 1, 1, 1]
linkCollisionShapeIndices = [joint_col, joint_col, joint_col, box_col, box_col, box_col]
linkVisualShapeIndices = [joint_vis, joint_vis, joint_vis, box_vis, box_vis, box_vis]
linkPositions = [[0, 0, 0], [0, 0, 0], [0, 0, 0], (0.0, -0.2747880220413208, 0.0), (0.0, -0.138947, 0.0), (2.980232238769531e-07, -2.384185791015625e-07, 0.1299997866153717)]
linkOrientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, -2.60770320892334e-08, 5.587935447692871e-09, 1.0)]
linkInertialFramePositions = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
linkParentIndices = [0, 1, 2, 3, 4, 5]
linkJointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_FIXED]
linkJointAxis = [[1, 0, 0], [0, 1, 0], [0, 0, 1], (-1.0, 0.0, 0.0), [0, 1, 0], [0, 0, 0]]

sphereUid = bc.createMultiBody(mass,
                              box_col,
                              box_vis,
                              basePosition,
                              baseOrientation,
                              linkMasses=linkMasses,
                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                              linkVisualShapeIndices=linkVisualShapeIndices,
                              linkPositions=linkPositions,
                              linkOrientations=linkOrientations,
                              linkInertialFramePositions=linkInertialFramePositions,
                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                              linkParentIndices=linkParentIndices,
                              linkJointTypes=linkJointTypes,
                              linkJointAxis=linkJointAxis)
print('getNumJoints:', bc.getNumJoints(sphereUid))

# tune joint damping
for i in range(bc.getNumJoints(sphereUid)):
  bc.changeDynamics(sphereUid, i, jointDamping=0.0001)

# set joint lower and upper limits
shoulder_min = bc.getEulerFromQuaternion([-0.397618, -0.885139, -0.759182, -0.014004])
shoulder_max = bc.getEulerFromQuaternion([0.654265, 0.959910, 0.718185, 0.643939])
elbow_min = 0.4
elbow_max = 2.5
print(shoulder_min)
print(shoulder_max)

bc.changeDynamics(sphereUid, 0, jointLowerLimit=shoulder_min[0], jointUpperLimit=shoulder_max[0])  # shoulder
bc.changeDynamics(sphereUid, 1, jointLowerLimit=shoulder_min[1], jointUpperLimit=shoulder_max[1])
bc.changeDynamics(sphereUid, 2, jointLowerLimit=shoulder_min[2], jointUpperLimit=shoulder_max[2])
bc.changeDynamics(sphereUid, 3, jointLowerLimit=elbow_min, jointUpperLimit=elbow_max)  # elbow
bc.changeDynamics(sphereUid, 4, jointLowerLimit=-1.57, jointUpperLimit=1.57)  # cp

for i in range(bc.getNumJoints(sphereUid)):
  print(i, 'link: ', bc.getLinkState(sphereUid, i))
  print(i, 'joint: ', bc.getJointInfo(sphereUid, i))


# get human arm joint positions
shoulder_joint = [-0.24918899093443897, -0.7468826816822298, 0.5814163450803189, 0.2050027811364538]
shoulder_joint = bc.getEulerFromQuaternion(shoulder_joint)
elbow_joint = 0.4544737022476803
print(shoulder_joint)

time.sleep(2)

# red, green, blue

bc.setJointMotorControl2(sphereUid, 0, p.POSITION_CONTROL, targetPosition=0.8, targetVelocity=0,
                            force=100,
                            positionGain=0.01,
                            velocityGain=0.01)
# bc.setJointMotorControl2(sphereUid, 1, p.POSITION_CONTROL, targetPosition=0.5, targetVelocity=0,
#                             force=100,
#                             positionGain=0.01,
#                             velocityGain=0.01)
bc.setJointMotorControl2(sphereUid, 2, p.POSITION_CONTROL, targetPosition=0.8, targetVelocity=0,
                            force=100,
                            positionGain=0.01,
                            velocityGain=0.01)
# bc.setJointMotorControl2(sphereUid, 3, p.POSITION_CONTROL, targetPosition=elbow_joint, targetVelocity=0,
#                             force=100,
#                             positionGain=0.01,
#                             velocityGain=0.01)

while True:
#   bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

#   print('link: ', bc.getLinkState(sphereUid, 0))

  # eef
#   bc.setJointMotorControl2(sphereUid, 4, p.POSITION_CONTROL, targetPosition=2.0)

  bc.stepSimulation()


Reset(humanoid)
s.disconnect()

