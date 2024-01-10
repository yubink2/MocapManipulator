import sys
sys.path.append("constraint")
from tsr import *
import numpy as np
from transformation import *
import pybullet as p


def pose_to_transform(pos, orn):
    translation_matrix = np.array([
        [1, 0, 0, pos[0]],
        [0, 1, 0, pos[1]],
        [0, 0, 1, pos[2]],
        [0, 0, 0, 1]
    ])

    rotation_matrix = quaternion_matrix(orn)

    transform_matrix = np.dot(translation_matrix, rotation_matrix)
    return transform_matrix

def transform_to_pose(transform_matrix):
    pos = transform_matrix[:3, 3]
    orn = quaternion_from_matrix(transform_matrix[:3, :3])
    return pos, orn


def define_tsrchain(env):
    # rotation of grasp pose about its elbow
    human_pos, human_orn = env.bc.getLinkState(env.humanoid._humanoid, 4)[4:6]   # elbow pose in world frame
    T0_w = pose_to_transform(human_pos, human_orn)
    Tw_e = np.eye(4)
    Tw_e[1][3] = -0.12
    Bw = np.zeros((6,2))
    Bw[5][:] = [0.4, 2.5]
    constraint1 = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw)

    # rotation of grasp pose about its shoulder
    shoulder_pos, shoulder_orn = env.bc.getLinkState(env.humanoid._humanoid, 3)[4:6]  # shoulder pose in world frame
    T0_w = pose_to_transform(shoulder_pos, shoulder_orn)
    Tw_e = np.eye(4)
    Tw_e[1][3] = -0.26
    # Tw_e[1][3] = -0.14*2 + -0.12
    Bw = np.zeros((6,2))
    shoulder_min = env.bc.getEulerFromQuaternion([-0.397618, -0.885139, -0.759182, -0.014004])
    shoulder_max = env.bc.getEulerFromQuaternion([0.654265, 0.959910, 0.718185, 0.643939])
    Bw[3][0] = shoulder_min[0]
    Bw[4][0] = shoulder_min[1]
    Bw[5][0] = shoulder_min[2]
    Bw[3][1] = shoulder_max[0]
    Bw[4][1] = shoulder_max[1]
    Bw[5][1] = shoulder_max[2]
    constraint2 = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw)

    # rotation of eef about grasp pose
    eef_pos, eef_orn = env.bc.getLinkState(env.robot.id, env.robot.eef_id)[4:6]  # eef pose in world frame
    pos_up = (human_pos[0], human_pos[1]+0.13, human_pos[2])
    T0_w = pose_to_transform(pos_up, env.bc.getQuaternionFromEuler([-1.57, 0.15, -1.57])) 
    # T0_w = pose_to_transform(pos_up, human_orn) 
    T0_e = pose_to_transform(eef_pos, eef_orn)
    Tw_e = np.linalg.inv(T0_w) @ T0_e  # desire pose of eef relative to grasp
    T0_w = np.eye(4)
    Bw = np.zeros((6,2))
    # Bw[3][:] = [-np.pi/8, np.pi/8]
    # Bw[4][:] = [-np.pi/8, np.pi/8]
    # Bw[5][:] = [-np.pi/8, np.pi/8]
    constraint3 = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw)

    # constraint applied over the whole trajectory
    tsrchain = TSRChain(sample_start=False, sample_goal=False, constrain=True, 
                                      TSRs = [constraint1, constraint2, constraint3])
    return tsrchain


def define_virtual_manip(env):
    shoulder_pos, shoulder_orn = env.bc.getLinkState(env.humanoid._humanoid, 3)[4:6]   
    elbow_pos, elbow_orn = env.bc.getLinkState(env.humanoid._humanoid, 4)[4:6]   
    wrist_pos, wrist_orn = env.bc.getLinkState(env.humanoid._humanoid, 5)[4:6]

    sphereRadius = 0.05
    visSphereId = env.bc_second.createVisualShape(p.GEOM_SPHERE, radius=sphereRadius)
    colSphereId = env.bc_second.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
    mass = 1
    visualShapeId = -1

    basePosition = shoulder_pos
    baseOrientation = shoulder_orn
    link_Masses = [1, 1]
    linkCollisionShapeIndices = [1, 2]
    linkVisualShapeIndices = [visSphereId, visSphereId]
    linkPositions = [elbow_pos, wrist_pos]
    linkOrientations = [elbow_orn, wrist_orn]
    linkInertialFramePositions = [[0, 0, 0], [0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
    indices = [0, 1]
    jointTypes = [p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]
    axis = [[0, 1, 0], [0, 1, 0]]

    virtualManipId = env.bc_second.createMultiBody(mass,
                                            colSphereId,
                                            visualShapeId,
                                            basePosition,
                                            baseOrientation,
                                            linkMasses=link_Masses,
                                            linkCollisionShapeIndices=linkCollisionShapeIndices,
                                            linkVisualShapeIndices=linkVisualShapeIndices,
                                            linkPositions=linkPositions,
                                            linkOrientations=linkOrientations,
                                            linkInertialFramePositions=linkInertialFramePositions,
                                            linkInertialFrameOrientations=linkInertialFrameOrientations,
                                            linkParentIndices=indices,
                                            linkJointTypes=jointTypes,
                                            linkJointAxis=axis)

    # dof?
    return virtualManipId