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
    # rotation of elbow about its shoulder
    shoulder_to_world = env.bc.getLinkState(env.humanoid._humanoid, 3)[4:6]
    shoulder_to_world_inertial = env.bc.getLinkState(env.humanoid._humanoid, 3)[2:4] 
    shoulder_joint_axis = [1, 1, 1]  # spherical
    elbow_to_shoulder = env.bc.getJointInfo(env.humanoid._humanoid, 4)[14:16]
    elbow_to_shoulder = env.bc.multiplyTransforms(shoulder_to_world_inertial[0], shoulder_to_world_inertial[1],
                                          elbow_to_shoulder[0], elbow_to_shoulder[1])
    
    T0_w = pose_to_transform(shoulder_to_world[0], shoulder_to_world[1])
    Tw_e = pose_to_transform(elbow_to_shoulder[0], elbow_to_shoulder[1])
    Bw = np.zeros((6,2))
    shoulder_min = env.bc.getEulerFromQuaternion([-0.397618, -0.885139, -0.759182, -0.014004])
    shoulder_max = env.bc.getEulerFromQuaternion([0.654265, 0.959910, 0.718185, 0.643939])
    Bw[3][0] = shoulder_min[0]
    Bw[4][0] = shoulder_min[1]
    Bw[5][0] = shoulder_min[2]
    Bw[3][1] = shoulder_max[0]
    Bw[4][1] = shoulder_max[1]
    Bw[5][1] = shoulder_max[2]
    constraint1 = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw, isSpherical = True, jointAxis = shoulder_joint_axis)

    # rotation of contact point about its elbow
    elbow_to_world = env.bc.getLinkState(env.humanoid._humanoid, 4)[4:6]
    elbow_joint_axis = env.bc.getJointInfo(env.humanoid._humanoid, 4)[13]
    wrist_to_elbow = env.bc.getJointInfo(env.humanoid._humanoid, 5)[14:16]
    cp_to_elbow = env.bc.multiplyTransforms(elbow_to_world[0], elbow_to_world[1],
                                          wrist_to_elbow[0], wrist_to_elbow[1])

    T0_w = pose_to_transform(elbow_to_world[0], elbow_to_world[1])
    Tw_e = pose_to_transform(cp_to_elbow[0], cp_to_elbow[1])
    Bw = np.zeros((6,2))
    elbow_min = 0.4
    elbow_max = 2.5
    Bw[5][:] = [elbow_min, elbow_max]
    constraint2 = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw, isSpherical = False, jointAxis = elbow_joint_axis)

    # # rotation of eef about contact point (QUESTION: how to figure out joint axis????)
    # cp_to_world = env.bc.multiplyTransforms(cp_to_elbow[0], cp_to_elbow[1],
    #                                     elbow_to_world[0], elbow_to_world[1])
    # eef_to_world = env.bc.getLinkState(env.robot.id, env.robot.eef_id)[4:6]
    
    # desired_eef_to_world_pos  = env.bc.getLinkState(env.humanoid._humanoid, 4)[0]
    # desired_eef_to_world_pos = [desired_eef_to_world_pos[0], desired_eef_to_world_pos[1]+0.13, desired_eef_to_world_pos[2]]
    # desired_eef_to_world_orn = env.bc.getQuaternionFromEuler([-1.57, 0.15, -1.57])  # NEED TO FIX THIS

    # world_to_cp = env.bc.invertTransform(cp_to_world[0], cp_to_world[1])
    # eef_to_cp = env.bc.multiplyTransforms(eef_to_world[0], eef_to_world[1],
    #                                     world_to_cp[0], world_to_cp[1])
    # desired_eef_to_cp = env.bc.multiplyTransforms(desired_eef_to_world_pos, desired_eef_to_world_orn,
    #                                             world_to_cp[0], world_to_cp[1])
    # cp_joint_axis = [0, 0, 0]  # assume fixed joint for now.. NEED TO FIX THIS

    # T0_w = pose_to_transform(cp_to_world[0], cp_to_world[1])
    # # Tw_e = pose_to_transform(eef_to_cp[0], eef_to_cp[1])
    # Tw_e = pose_to_transform(desired_eef_to_cp[0], desired_eef_to_cp[1])
    # Bw = np.zeros((6,2))  # ???
    # constraint3 = TSR(T0_w = T0_w, Tw_e = Tw_e, Bw = Bw, isSpherical = False, jointAxis = cp_joint_axis)

    # constraint applied over the whole trajectory
    tsrchain = TSRChain(sample_start=False, sample_goal=False, constrain=True, 
                                      TSRs = [constraint1, constraint2])
    return tsrchain


def define_virtual_manip(env, tsrchain):
    basePosition = []
    baseOrientation = []
    linkMasses = []
    linkCollisionShapeIndices = []
    linkVisualShapeIndices = []
    linkPositions = []
    linkOrientations = []
    linkInertialFramePositions = []
    linkInertialFrameOrientations = []
    parentIndices = []
    jointTypes = []
    axis = []

    box_vis = -1
    box_col = env.bc.createCollisionShape(p.GEOM_BOX,halfExtents=[0.05, 0.05, 0.05])
    mass = 1
    joint_col = -1
    joint_vis = -1
    joint_pos = [0, 0, 0]
    joint_orn = [0, 0, 0, 1]
    linkInertialFramePos = [0, 0, 0]
    linkInertialFrameOrn = [0, 0, 0, 1]

    virtualManipDOF = 0

    shoulder_to_world = env.bc.getLinkState(env.humanoid._humanoid, 3)[4:6]
    basePosition = shoulder_to_world[0]
    baseOrientation = shoulder_to_world[1]

    # shoulder - joint 1, 2, 3
    linkMasses.append(mass)
    linkCollisionShapeIndices.append(joint_col)
    linkVisualShapeIndices.append(joint_vis)
    linkPositions.append(joint_pos)
    linkOrientations.append(joint_orn)
    linkInertialFramePositions.append(linkInertialFramePos)
    linkInertialFrameOrientations.append(linkInertialFrameOrn)
    parentIndices.append(0)
    jointTypes.append(p.JOINT_REVOLUTE)
    axis.append([1, 0, 0])

    linkMasses.append(mass)
    linkCollisionShapeIndices.append(joint_col)
    linkVisualShapeIndices.append(joint_vis)
    linkPositions.append(joint_pos)
    linkOrientations.append(joint_orn)
    linkInertialFramePositions.append(linkInertialFramePos)
    linkInertialFrameOrientations.append(linkInertialFrameOrn)
    parentIndices.append(1)
    jointTypes.append(p.JOINT_REVOLUTE)
    axis.append([0, 1, 0])

    # elbow from shoulder
    shoulder_to_world_inertial = env.bc.getLinkState(env.humanoid._humanoid, 3)[2:4] 
    elbow_to_world = env.bc.getLinkState(env.humanoid._humanoid, 4)[4:6]
    elbow_to_shoulder = env.bc.getJointInfo(env.humanoid._humanoid, 4)[14:16]
    elbow_to_shoulder = env.bc.multiplyTransforms(shoulder_to_world_inertial[0], shoulder_to_world_inertial[1],
                                          elbow_to_shoulder[0], elbow_to_shoulder[1])

    linkMasses.append(mass)
    linkCollisionShapeIndices.append(box_col)
    linkVisualShapeIndices.append(box_vis)
    linkPositions.append(elbow_to_shoulder[0])
    linkOrientations.append(elbow_to_shoulder[1])
    linkInertialFramePositions.append(linkInertialFramePos)
    linkInertialFrameOrientations.append(linkInertialFrameOrn)
    parentIndices.append(2)
    jointTypes.append(p.JOINT_REVOLUTE)
    axis.append([0, 0, 1])

    virtualManipDOF += 3

    # cp from elbow
    elbow_joint_axis = env.bc.getJointInfo(env.humanoid._humanoid, 4)[13]
    cp_to_elbow = env.bc.getJointInfo(env.humanoid._humanoid, 5)[14:16]

    linkMasses.append(mass)
    linkCollisionShapeIndices.append(box_col)
    linkVisualShapeIndices.append(box_vis)
    linkPositions.append(cp_to_elbow[0])
    linkOrientations.append(cp_to_elbow[1])
    linkInertialFramePositions.append(linkInertialFramePos)
    linkInertialFrameOrientations.append(linkInertialFrameOrn)
    parentIndices.append(3)
    jointTypes.append(p.JOINT_REVOLUTE)
    axis.append(elbow_joint_axis)

    virtualManipDOF += 1

    # eef from cp
    desired_eef_to_world_pos = env.bc.getLinkState(env.humanoid._humanoid, 4)[0]
    desired_eef_to_world_pos = [desired_eef_to_world_pos[0], desired_eef_to_world_pos[1], desired_eef_to_world_pos[2]+0.13]
    desired_eef_to_world_orn = env.bc.getLinkState(env.humanoid._humanoid, 4)[1]

    cp_to_world = env.bc.multiplyTransforms(cp_to_elbow[0], cp_to_elbow[1],
                                        elbow_to_world[0], elbow_to_world[1])
    cp_to_world = env.bc.getLinkState(env.humanoid._humanoid, 4)[:2]
    world_to_cp = env.bc.invertTransform(cp_to_world[0], cp_to_world[1])
    desired_eef_to_cp = env.bc.multiplyTransforms(desired_eef_to_world_pos, desired_eef_to_world_orn,
                                                world_to_cp[0], world_to_cp[1])

    linkMasses.append(mass)
    linkCollisionShapeIndices.append(box_col)
    linkVisualShapeIndices.append(box_vis)
    linkPositions.append(desired_eef_to_cp[0])
    linkOrientations.append(desired_eef_to_cp[1])
    linkInertialFramePositions.append(linkInertialFramePos)
    linkInertialFrameOrientations.append(linkInertialFrameOrn)
    parentIndices.append(4)
    jointTypes.append(p.JOINT_REVOLUTE)
    axis.append([1, 0, 0])  # try both x and y axis

    virtualManipDOF += 1

    # create virtual manip
    virtualManipId = env.bc.createMultiBody(mass,
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
                              linkParentIndices=parentIndices,
                              linkJointTypes=jointTypes,
                              linkJointAxis=axis)

    print("*basePosition: ", basePosition)
    print("*baseOrientation: ", baseOrientation)
    print("*linkMasses: ", linkMasses)
    print("*linkCollisionShapeIndices: ", linkCollisionShapeIndices)
    print("*linkVisualShapeIndices: ", linkVisualShapeIndices)
    print("*linkPositions: ", linkPositions)
    print("*linkOrientations: ", linkOrientations)
    print("*linkInertialFramePositions: ", linkInertialFramePositions)
    print("*linkInertialFrameOrientations: ", linkInertialFrameOrientations)
    print("*linkParentIndices: ", parentIndices)
    print("*linkJointTypes: ", jointTypes)
    print("*linkJointAxis: ", axis)

    # set joint limits
    shoulder_min = env.bc.getEulerFromQuaternion([-0.397618, -0.885139, -0.759182, -0.014004])
    shoulder_max = env.bc.getEulerFromQuaternion([0.654265, 0.959910, 0.718185, 0.643939])
    elbow_min = 0.4
    elbow_max = 2.5

    env.bc.changeDynamics(virtualManipId, 0, jointLowerLimit=shoulder_min[0], jointUpperLimit=shoulder_max[0])  # shoulder
    env.bc.changeDynamics(virtualManipId, 1, jointLowerLimit=shoulder_min[1], jointUpperLimit=shoulder_max[1])
    env.bc.changeDynamics(virtualManipId, 2, jointLowerLimit=shoulder_min[2], jointUpperLimit=shoulder_max[2])
    env.bc.changeDynamics(virtualManipId, 3, jointLowerLimit=elbow_min, jointUpperLimit=elbow_max)  # elbow
    env.bc.changeDynamics(virtualManipId, 4, jointLowerLimit=-1.57, jointUpperLimit=1.57)  # cp

    return virtualManipId, virtualManipDOF


def define_virtual_manip_dummies(env, tsrchain):
    basePosition = []
    baseOrientation = []
    linkMasses = []
    linkCollisionShapeIndices = []
    linkVisualShapeIndices = []
    linkPositions = []
    linkOrientations = []
    linkInertialFramePositions = []
    linkInertialFrameOrientations = []
    parentIndices = []
    jointTypes = []
    axis = []

    box_vis = -1
    box_col = env.bc.createCollisionShape(p.GEOM_BOX,halfExtents=[0.06, 0.06, 0.06])
    mass = 1
    joint_col = -1
    joint_vis = -1
    joint_pos = [0, 0, 0]
    joint_orn = [0, 0, 0, 1]
    linkInertialFramePos = [0, 0, 0]
    linkInertialFrameOrn = [0, 0, 0, 1]

    virtualManipDOF = 0
    isBase = False
    cnt = 0

    for i, tsr in enumerate(tsrchain.TSRs):
        if (i == 0):
            pos, orn = transform_to_pose(tsr.T0_w)
            basePosition = pos
            baseOrientation = pos
            isBase = True
        else:
            isBase = False
 
        if (tsr.isSpherical):
            for _ in range(3):
                linkMasses.append(mass)
                linkCollisionShapeIndices.append(joint_col)
                linkVisualShapeIndices.append(joint_vis)
                linkPositions.append(joint_pos)
                linkOrientations.append(joint_orn)
                linkInertialFramePositions.append(linkInertialFramePos)
                linkInertialFrameOrientations.append(linkInertialFrameOrn)
                parentIndices.append(i+cnt)
                jointTypes.append(p.JOINT_REVOLUTE)
                cnt += 1

            axis.append([1, 0, 0])
            axis.append([0, 1, 0])
            axis.append([0, 0, 1])
            virtualManipDOF += 3

        if (not isBase):
            pos, orn = transform_to_pose(tsr.Tw_e)
            linkMasses.append(mass)
            linkCollisionShapeIndices.append(box_col)
            linkVisualShapeIndices.append(box_vis)
            linkPositions.append(pos)
            linkOrientations.append(orn)
            linkInertialFramePositions.append(linkInertialFramePos)
            linkInertialFrameOrientations.append(linkInertialFrameOrn)
            parentIndices.append(i+cnt-1)
            jointTypes.append(p.JOINT_REVOLUTE)
            axis.append(tsr.jointAxis)
            
            virtualManipDOF += 1

    print("*basePosition: ", basePosition)
    print("*baseOrientation: ", baseOrientation)
    print("*linkMasses: ", linkMasses)
    print("*linkCollisionShapeIndices: ", linkCollisionShapeIndices)
    print("*linkVisualShapeIndices: ", linkVisualShapeIndices)
    print("*linkPositions: ", linkPositions)
    print("*linkOrientations: ", linkOrientations)
    print("*linkInertialFramePositions: ", linkInertialFramePositions)
    print("*linkInertialFrameOrientations: ", linkInertialFrameOrientations)
    print("*linkParentIndices: ", parentIndices)
    print("*linkJointTypes: ", jointTypes)
    print("*linkJointAxis: ", axis)
    print("*virtualManipDOF: ", virtualManipDOF)
    
    virtualManipId = env.bc.createMultiBody(mass,
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
                              linkParentIndices=parentIndices,
                              linkJointTypes=jointTypes,
                              linkJointAxis=axis)
    
    return virtualManipId, virtualManipDOF

