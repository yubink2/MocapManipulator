import os, inspect
import math
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from pybullet_utils.bullet_client import BulletClient
import pybullet_data

jointTypes = [
    "JOINT_REVOLUTE", "JOINT_PRISMATIC", "JOINT_SPHERICAL", "JOINT_PLANAR", "JOINT_FIXED"
]


class HumanoidPose(object):

  def __init__(self):
    pass

  def Reset(self):

    self._basePos = [0, 0, 0]
    self._baseLinVel = [0, 0, 0]
    self._baseOrn = [0, 0, 0, 1]
    self._baseAngVel = [0, 0, 0]

    self._chestRot = [0, 0, 0, 1]
    self._chestVel = [0, 0, 0]
    self._neckRot = [0, 0, 0, 1]
    self._neckVel = [0, 0, 0]

    self._rightHipRot = [0, 0, 0, 1]
    self._rightHipVel = [0, 0, 0]
    self._rightKneeRot = [0]
    self._rightKneeVel = [0]
    self._rightAnkleRot = [0, 0, 0, 1]
    self._rightAnkleVel = [0, 0, 0]

    self._rightShoulderRot = [0, 0, 0, 1]
    self._rightShoulderVel = [0, 0, 0]
    self._rightElbowRot = [0]
    self._rightElbowVel = [0]

    self._leftHipRot = [0, 0, 0, 1]
    self._leftHipVel = [0, 0, 0]
    self._leftKneeRot = [0]
    self._leftKneeVel = [0]
    self._leftAnkleRot = [0, 0, 0, 1]
    self._leftAnkleVel = [0, 0, 0]

    self._leftShoulderRot = [0, 0, 0, 1]
    self._leftShoulderVel = [0, 0, 0]
    self._leftElbowRot = [0]
    self._leftElbowVel = [0]

  def ComputeLinVel(self, posStart, posEnd, deltaTime):
    vel = [(posEnd[0] - posStart[0]) / deltaTime, (posEnd[1] - posStart[1]) / deltaTime,
           (posEnd[2] - posStart[2]) / deltaTime]
    return vel

  def ComputeAngVel(self, ornStart, ornEnd, deltaTime, bullet_client):
    dorn = bullet_client.getDifferenceQuaternion(ornStart, ornEnd)
    axis, angle = bullet_client.getAxisAngleFromQuaternion(dorn)
    angVel = [(axis[0] * angle) / deltaTime, (axis[1] * angle) / deltaTime,
              (axis[2] * angle) / deltaTime]
    return angVel

  def NormalizeQuaternion(self, orn):
    length2 = orn[0] * orn[0] + orn[1] * orn[1] + orn[2] * orn[2] + orn[3] * orn[3]
    if (length2 > 0):
      length = math.sqrt(length2)
    #print("Normalize? length=",length)

  def PostProcessMotionData(self, frameData):
    baseOrn1Start = [frameData[5], frameData[6], frameData[7], frameData[4]]
    self.NormalizeQuaternion(baseOrn1Start)
    chestRotStart = [frameData[9], frameData[10], frameData[11], frameData[8]]

    neckRotStart = [frameData[13], frameData[14], frameData[15], frameData[12]]
    rightHipRotStart = [frameData[17], frameData[18], frameData[19], frameData[16]]
    rightAnkleRotStart = [frameData[22], frameData[23], frameData[24], frameData[21]]
    rightShoulderRotStart = [frameData[26], frameData[27], frameData[28], frameData[25]]
    leftHipRotStart = [frameData[31], frameData[32], frameData[33], frameData[30]]
    leftAnkleRotStart = [frameData[36], frameData[37], frameData[38], frameData[35]]
    leftShoulderRotStart = [frameData[40], frameData[41], frameData[42], frameData[39]]

  def Slerp(self, frameFraction, frameData, frameDataNext, bullet_client):
    keyFrameDuration = frameData[0]
    basePos1Start = [frameData[1], frameData[2], frameData[3]]
    basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
    self._basePos = [
        basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
        basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
        basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
    ]
    self._baseLinVel = self.ComputeLinVel(basePos1Start, basePos1End, keyFrameDuration)
    baseOrn1Start = [frameData[5], frameData[6], frameData[7], frameData[4]]
    baseOrn1Next = [frameDataNext[5], frameDataNext[6], frameDataNext[7], frameDataNext[4]]

    self._baseOrn = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
    self._baseAngVel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, keyFrameDuration,
                                          bullet_client)
    self._basePos = [0, 0, 0]
    self._baseLinVel = [0, 0, 0]
    self._baseOrn = bullet_client.getQuaternionFromEuler((0, 1.57, 0))  # TODO
    self._baseAngVel = [0, 0, 0]

    # ##### pre-rotate to make z-up
    # basePos1 = [
    #     basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
    #     basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
    #     basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
    # ]
    # baseOrn1 = bullet_client.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
    # y2zPos=[0,0,0.0]
    # y2zOrn = bullet_client.getQuaternionFromEuler([1.57,0,0])
    # basePos,baseOrn = bullet_client.multiplyTransforms(y2zPos, y2zOrn,basePos1,baseOrn1)
    # #####

    chestRotStart = [frameData[9], frameData[10], frameData[11], frameData[8]]
    chestRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11], frameDataNext[8]]
    self._chestRot = bullet_client.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)
    self._chestVel = self.ComputeAngVel(chestRotStart, chestRotEnd, keyFrameDuration,
                                       bullet_client)
    self._chestRot = [0, 0, 0, 1]
    self._chestVel = [0, 0, 0]

    neckRotStart = [frameData[13], frameData[14], frameData[15], frameData[12]]
    neckRotEnd = [frameDataNext[13], frameDataNext[14], frameDataNext[15], frameDataNext[12]]
    self._neckRot = bullet_client.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)
    self._neckVel = self.ComputeAngVel(neckRotStart, neckRotEnd, keyFrameDuration, bullet_client)
    self._neckRot = [0, 0, 0, 1]
    self._neckVel = [0, 0, 0]

    rightHipRotStart = [frameData[17], frameData[18], frameData[19], frameData[16]]
    rightHipRotEnd = [frameDataNext[17], frameDataNext[18], frameDataNext[19], frameDataNext[16]]
    self._rightHipRot = bullet_client.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd,
                                                         frameFraction)
    self._rightHipVel = self.ComputeAngVel(rightHipRotStart, rightHipRotEnd, keyFrameDuration,
                                           bullet_client)

    rightKneeRotStart = [frameData[20]]
    rightKneeRotEnd = [frameDataNext[20]]
    self._rightKneeRot = [
        rightKneeRotStart[0] + frameFraction * (rightKneeRotEnd[0] - rightKneeRotStart[0])
    ]
    self._rightKneeVel = [(rightKneeRotEnd[0] - rightKneeRotStart[0]) / keyFrameDuration]

    rightAnkleRotStart = [frameData[22], frameData[23], frameData[24], frameData[21]]
    rightAnkleRotEnd = [frameDataNext[22], frameDataNext[23], frameDataNext[24], frameDataNext[21]]
    self._rightAnkleRot = bullet_client.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd,
                                                           frameFraction)
    self._rightAnkleVel = self.ComputeAngVel(rightAnkleRotStart, rightAnkleRotEnd,
                                             keyFrameDuration, bullet_client)

    ### TODO
    rightShoulderRotStart = [frameData[26], frameData[27], frameData[28], frameData[25]]
    rightShoulderRotEnd = [
        frameDataNext[26], frameDataNext[27], frameDataNext[28], frameDataNext[25]
    ]
    self._rightShoulderRot = bullet_client.getQuaternionSlerp(rightShoulderRotStart,
                                                              rightShoulderRotEnd, frameFraction)
    self._rightShoulderVel = self.ComputeAngVel(rightShoulderRotStart, rightShoulderRotEnd,
                                                keyFrameDuration, bullet_client)
    # self._rightShoulderRot = [0, 1, 0, 0]
    # self._rightShoulderVel = [0, 0, 0]

    # right elbow 
    rightElbowRotStart = [frameData[29]]
    rightElbowRotEnd = [frameDataNext[29]]
    self._rightElbowRot = [
        rightElbowRotStart[0] + frameFraction * (rightElbowRotEnd[0] - rightElbowRotStart[0])
    ]
    self._rightElbowVel = [(rightElbowRotEnd[0] - rightElbowRotStart[0]) / keyFrameDuration]
    # self._rightElbowRot = [0]
    # self._rightElbowVel = [0]

    leftHipRotStart = [frameData[31], frameData[32], frameData[33], frameData[30]]
    leftHipRotEnd = [frameDataNext[31], frameDataNext[32], frameDataNext[33], frameDataNext[30]]
    self._leftHipRot = bullet_client.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd,
                                                        frameFraction)
    self._leftHipVel = self.ComputeAngVel(leftHipRotStart, leftHipRotEnd, keyFrameDuration,
                                          bullet_client)

    leftKneeRotStart = [frameData[34]]
    leftKneeRotEnd = [frameDataNext[34]]
    self._leftKneeRot = [
        leftKneeRotStart[0] + frameFraction * (leftKneeRotEnd[0] - leftKneeRotStart[0])
    ]
    self._leftKneeVel = [(leftKneeRotEnd[0] - leftKneeRotStart[0]) / keyFrameDuration]

    leftAnkleRotStart = [frameData[36], frameData[37], frameData[38], frameData[35]]
    leftAnkleRotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38], frameDataNext[35]]
    self._leftAnkleRot = bullet_client.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd,
                                                          frameFraction)
    self._leftAnkleVel = self.ComputeAngVel(leftAnkleRotStart, leftAnkleRotEnd, keyFrameDuration,
                                            bullet_client)

    # # prohibits left lower body movements
    # self._leftHipRot = [0, 0, 0, 1]
    # self._leftHipVel = [0, 0, 0]
    # self._leftKneeRot = [0]
    # self._leftKneeVel = [0]
    # self._leftAnkleRot = [0, 0, 1, 0]
    # self._leftAnkleVel = [0, 0, 0]

    # # prohibits right lower body movements
    # self._rightHipRot = [0, 0, 0, 1]
    # self._rightHipVel = [0, 0, 0]
    # self._rightKneeRot = [0]
    # self._rightKneeVel = [0]
    # self._rightAnkleRot = [0, 0, 1, 0]
    # self._rightAnkleVel = [0, 0, 0]

    leftShoulderRotStart = [frameData[40], frameData[41], frameData[42], frameData[39]]
    leftShoulderRotEnd = [
        frameDataNext[40], frameDataNext[41], frameDataNext[42], frameDataNext[39]
    ]
    self._leftShoulderRot = bullet_client.getQuaternionSlerp(leftShoulderRotStart,
                                                             leftShoulderRotEnd, frameFraction)
    self._leftShoulderVel = self.ComputeAngVel(leftShoulderRotStart, leftShoulderRotEnd,
                                               keyFrameDuration, bullet_client)
    # self._leftShoulderRot = [0, 1, 0, 0]
    # self._leftShoulderVel = [0, 0, 0]

    leftElbowRotStart = [frameData[43]]
    leftElbowRotEnd = [frameDataNext[43]]
    self._leftElbowRot = [
        leftElbowRotStart[0] + frameFraction * (leftElbowRotEnd[0] - leftElbowRotStart[0])
    ]
    self._leftElbowVel = [(leftElbowRotEnd[0] - leftElbowRotStart[0]) / keyFrameDuration]
    # self._leftElbowRot = [0]
    # self._leftElbowVel = [0]


class Humanoid(object):

  def __init__(self, pybullet_client, motion_data, baseShift, ornShift=[0,0,0,1]):
    """Constructs a humanoid and reset it to the initial states.
    Args:
      pybullet_client: The instance of BulletClient to manage different
        simulations.
    """
    self._baseShift = baseShift
    self._ornShift = ornShift
    self._pybullet_client = pybullet_client

    self.kin_client = BulletClient(
        pybullet_client.DIRECT
    ) 
    self.kin_client.resetSimulation()
    self.kin_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.kin_client.setGravity(0, 0, -9.8)

    self._motion_data = motion_data
    print("LOADING humanoid!")
    
    # useFixedBase=False if want to enable base motion (e.g., walking forward) 

    self._humanoid = self._pybullet_client.loadURDF("urdf/humanoid_with_rev.urdf", [0, 0.9, 0],
                                                    globalScaling=0.22,
                                                    useFixedBase=True)

    self._kinematicHumanoid = self.kin_client.loadURDF("urdf/humanoid_with_rev.urdf", [0, 0.9, 0],
                                                       globalScaling=0.22,
                                                       useFixedBase=True)

    # self._humanoid = self._pybullet_client.loadURDF("urdf/humanoid_with_rev_scaled.urdf", [0, 0.9, 0],
    #                                                 useFixedBase=True)

    # self._kinematicHumanoid = self.kin_client.loadURDF("urdf/humanoid_with_rev_scaled.urdf", [0, 0.9, 0],
    #                                                    useFixedBase=True)

    pose = HumanoidPose()

    for i in range(self._motion_data.NumFrames() - 1):
      frameData = self._motion_data._motion_data['Frames'][i]
      pose.PostProcessMotionData(frameData)

    self._pybullet_client.resetBasePositionAndOrientation(self._humanoid, self._baseShift, self._ornShift)
    self._pybullet_client.changeDynamics(self._humanoid, -1, linearDamping=0, angularDamping=0)
    
    # change colors of the human model limbs
    humanoid_color = [255/255, 160/255, 45/255, 1] # [255/255, 192/255, 0/255, 1]
    # bed: 0.92, 0.9, 0.88
    for j in range(self._pybullet_client.getNumJoints(self._humanoid)):  
      ji = self._pybullet_client.getJointInfo(self._humanoid, j)
      self._pybullet_client.changeDynamics(self._humanoid, j, linearDamping=0, angularDamping=0)
      self._pybullet_client.changeVisualShape(self._humanoid, j, rgbaColor=humanoid_color)

    self._initial_state = self._pybullet_client.saveState()
    self._allowed_body_parts = [11, 14]
    self.Reset()
    
    self._contact_point = [0, 0, 0, 0]
    self._rightShoulderJointAnglesList = []
    self._rightElbowJointAnglesList = []

  def Reset(self):
    self._pybullet_client.restoreState(self._initial_state)
    self.SetSimTime(0)
    pose = self.InitializePoseFromMotionData()
    self.ApplyPose(pose, True, True, self._humanoid, self._pybullet_client)

  def RenderReference(self, t, bc):
    self.SetSimTime(t)
    frameData = self._motion_data._motion_data['Frames'][self._frame]
    frameDataNext = self._motion_data._motion_data['Frames'][self._frameNext]
    pose = HumanoidPose()
    pose.Slerp(self._frameFraction, frameData, frameDataNext, self._pybullet_client)
    # print('renderref--rightShoulderRot: ', pose._rightShoulderRot, " vel: ", pose._rightShoulderVel)
    # print('in RenderReference() ', pose._rightElbowRot)
    
    self._rightShoulderJointAnglesList.append(bc.getEulerFromQuaternion(pose._rightShoulderRot))
    self._rightElbowJointAnglesList.append(pose._rightElbowRot)
    self.ApplyPose(pose, True, True, self._humanoid, self._pybullet_client)

  def RenderReferenceWithoutApplyPose(self, t):
    self.SetSimTime(t)
    frameData = self._motion_data._motion_data['Frames'][self._frame]
    frameDataNext = self._motion_data._motion_data['Frames'][self._frameNext]
    pose = HumanoidPose()
    pose.Slerp(self._frameFraction, frameData, frameDataNext, self._pybullet_client)
    # print('without--rightShoulderRot: ', pose._rightShoulderRot, " vel: ", pose._rightShoulderVel)
    return pose

  def CalcCycleCount(self, simTime, cycleTime):
    phases = simTime / cycleTime
    count = math.floor(phases)
    loop = True
    #count = (loop) ? count : cMathUtil::Clamp(count, 0, 1);
    return count

  def SetSimTime(self, t):
    self._simTime = t
    #print("SetTimeTime time =",t)
    keyFrameDuration = self._motion_data.KeyFrameDuraction()
    cycleTime = keyFrameDuration * (self._motion_data.NumFrames() - 1)
    #print("self._motion_data.NumFrames()=",self._motion_data.NumFrames())
    #print("cycleTime=",cycleTime)
    cycles = self.CalcCycleCount(t, cycleTime)
    #print("cycles=",cycles)
    frameTime = t - cycles * cycleTime
    if (frameTime < 0):
      frameTime += cycleTime

    #print("keyFrameDuration=",keyFrameDuration)
    #print("frameTime=",frameTime)
    self._frame = int(frameTime / keyFrameDuration)
    #print("self._frame=",self._frame)

    self._frameNext = self._frame + 1
    if (self._frameNext >= self._motion_data.NumFrames()):
      self._frameNext = self._frame

    self._frameFraction = (frameTime - self._frame * keyFrameDuration) / (keyFrameDuration)
    #print("self._frameFraction=",self._frameFraction)

  def Terminates(self):
    #check if any non-allowed body part hits the ground
    terminates = False
    pts = self._pybullet_client.getContactPoints()
    for p in pts:
      part = -1
      if (p[1] == self._humanoid):
        part = p[3]
      if (p[2] == self._humanoid):
        part = p[4]
      if (part >= 0 and part not in self._allowed_body_parts):
        terminates = True

    return terminates

  def BuildHeadingTrans(self, rootOrn):
    #align root transform 'forward' with world-space x axis
    eul = self._pybullet_client.getEulerFromQuaternion(rootOrn)
    refDir = [1, 0, 0]
    rotVec = self._pybullet_client.rotateVector(rootOrn, refDir)
    heading = math.atan2(-rotVec[2], rotVec[0])
    heading2 = eul[1]
    #print("heading=",heading)
    headingOrn = self._pybullet_client.getQuaternionFromAxisAngle([0, 1, 0], -heading)
    return headingOrn

  def GetPhase(self):
    keyFrameDuration = self._motion_data.KeyFrameDuraction()
    cycleTime = keyFrameDuration * (self._motion_data.NumFrames() - 1)
    phase = self._simTime / cycleTime
    phase = math.fmod(phase, 1.0)
    if (phase < 0):
      phase += 1
    return phase

  def BuildOriginTrans(self):
    rootPos, rootOrn = self._pybullet_client.getBasePositionAndOrientation(self._humanoid)

    #print("rootPos=",rootPos, " rootOrn=",rootOrn)
    invRootPos = [-rootPos[0], 0, -rootPos[2]]
    #invOrigTransPos, invOrigTransOrn = self._pybullet_client.invertTransform(rootPos,rootOrn)
    headingOrn = self.BuildHeadingTrans(rootOrn)
    #print("headingOrn=",headingOrn)
    headingMat = self._pybullet_client.getMatrixFromQuaternion(headingOrn)
    #print("headingMat=",headingMat)
    #dummy, rootOrnWithoutHeading = self._pybullet_client.multiplyTransforms([0,0,0],headingOrn, [0,0,0], rootOrn)
    #dummy, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0,0,0],rootOrnWithoutHeading, invOrigTransPos, invOrigTransOrn)

    invOrigTransPos, invOrigTransOrn = self._pybullet_client.multiplyTransforms([0, 0, 0],
                                                                                headingOrn,
                                                                                invRootPos,
                                                                                [0, 0, 0, 1])
    #print("invOrigTransPos=",invOrigTransPos)
    #print("invOrigTransOrn=",invOrigTransOrn)
    invOrigTransMat = self._pybullet_client.getMatrixFromQuaternion(invOrigTransOrn)
    #print("invOrigTransMat =",invOrigTransMat )
    return invOrigTransPos, invOrigTransOrn

  def InitializePoseFromMotionData(self):
    frameData = self._motion_data._motion_data['Frames'][self._frame]
    frameDataNext = self._motion_data._motion_data['Frames'][self._frameNext]
    pose = HumanoidPose()
    pose.Slerp(self._frameFraction, frameData, frameDataNext, self._pybullet_client)
    return pose

  def ApplyPose(self, pose, initializeBase, initializeVelocities, humanoid, bc):
    #todo: get tunable parametes from a json file or from URDF (kd, maxForce)
    if (initializeBase):
      bc.changeVisualShape(humanoid, 2, rgbaColor=[1, 0, 0, 1])
      basePos = [
          pose._basePos[0] + self._baseShift[0], pose._basePos[1] + self._baseShift[1],
          pose._basePos[2] + self._baseShift[2]
      ]

      bc.resetBasePositionAndOrientation(humanoid, basePos, pose._baseOrn)
      if initializeVelocities:
        bc.resetBaseVelocity(humanoid, pose._baseLinVel, pose._baseAngVel)
        #print("resetBaseVelocity=",pose._baseLinVel)

    else:
      bc.changeVisualShape(humanoid, 2, rgbaColor=[1, 1, 1, 1])

    # kp = 0.03
    # # chest = 1
    # # neck = 2
    # rightShoulder = 3
    # rightElbow = 4
    # # leftShoulder = 6
    # # leftElbow = 7
    # # rightHip = 9
    # # rightKnee = 10
    # # rightAnkle = 11
    # # leftHip = 12
    # # leftKnee = 13
    # # leftAnkle = 14
    # controlMode = bc.POSITION_CONTROL

    kp = 0.03
    right_shoulder_r = 3
    right_shoulder_p = 4
    right_shoulder_y = 5
    right_elbow = 7
    controlMode = bc.POSITION_CONTROL

    # kp = 0.03
    # chest = 1
    # neck = 2
    # right_shoulder_y = 3
    # right_shoulder_p = 4
    # right_shoulder_r = 5
    # right_elbow = 7
    # controlMode = bc.POSITION_CONTROL

    right_shoulder_ypr = bc.getEulerFromQuaternion(pose._rightShoulderRot)
    
    if (initializeBase):
      if initializeVelocities:

        bc.resetJointState(humanoid, right_shoulder_y, right_shoulder_ypr[0],
                          pose._rightShoulderVel[0])
        bc.resetJointState(humanoid, right_shoulder_p, right_shoulder_ypr[1],
                          pose._rightShoulderVel[1])
        bc.resetJointState(humanoid, right_shoulder_r, right_shoulder_ypr[2],
                          pose._rightShoulderVel[2])      
        bc.resetJointState(humanoid, right_elbow, pose._rightElbowRot[0], pose._rightElbowVel[0])
      else:

        bc.resetJointState(humanoid, right_shoulder_y, right_shoulder_ypr[0])
        bc.resetJointState(humanoid, right_shoulder_p, right_shoulder_ypr[1])
        bc.resetJointState(humanoid, right_shoulder_r, right_shoulder_ypr[2])
        bc.resetJointState(humanoid, right_elbow, pose._rightElbowRot[0])

    bc.setJointMotorControl2(humanoid,
                            right_shoulder_y,
                            controlMode,
                            targetPosition=right_shoulder_ypr[0],
                            positionGain=kp,
                            force=100)
    bc.setJointMotorControl2(humanoid,
                            right_shoulder_p,
                            controlMode,
                            targetPosition=right_shoulder_ypr[1],
                            positionGain=kp,
                            force=100)
    bc.setJointMotorControl2(humanoid,
                            right_shoulder_r,
                            controlMode,
                            targetPosition=right_shoulder_ypr[2],
                            positionGain=kp,
                            force=100)
    
    bc.setJointMotorControl2(humanoid,
                            right_elbow,
                            controlMode,
                            targetPosition=pose._rightElbowRot[0],
                            positionGain=kp,
                            force=60)
  
    # print('--applypose jointstate: ', bc.getJointState(self._humanoid, 4))    

    #debug space
    # if (False):
    #  for j in range (bc.getNumJoints(self._humanoid)):
    #    js = bc.getJointState(self._humanoid, j)
    #    print('joint j state in applypose: ', js)
      #  bc.resetJointState(self._humanoidDebug, j,js[0])
      #  jsm = bc.getJointStateMultiDof(self._humanoid, j)
      #  if (len(jsm[0])>0):
      #    bc.resetJointStateMultiDof(self._humanoidDebug,j,jsm[0])

  def GetBasePosition(self):
    pos, orn = self._pybullet_client.getBasePositionAndOrientation(self._humanoid)
    return pos
