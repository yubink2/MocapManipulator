import pybullet as p
import time
import numpy as np
    

p.connect(p.GUI)
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0,0)


#Sat part shapes
Sat_body = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2.0  , 2.3, 3.0])
Panel_right = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2, 0.2, 3])
Panel_left = p.createCollisionShape(p.GEOM_BOX,halfExtents=[2, 0.2, 3])

#The with other shapes linked to it
body_Mass = 500
visualShapeId = -1
link_Masses=[10, 10, 10, 10]



linkCollisionShapeIndices=[Panel_right, Panel_right, Panel_left, Panel_left]
nlnk=len(link_Masses)
linkVisualShapeIndices=[-1]*nlnk    #=[-1,-1,-1, ... , -1]
#link positions wrt the link they are attached to
BasePositionR_body = [4.2,0,0]
BasePositionR_panel = [4.2,0,0]
BasePositionL_body = [-4.2,0,0]
BasePositionL_panel = [-4.2,0,0]

linkPositions=[BasePositionR_body, BasePositionR_panel, BasePositionL_body, BasePositionL_panel,]
linkOrientations=[[0,0,0,1]]*nlnk
linkInertialFramePositions=[[0,0,0]]*nlnk

#linkInertialFramePositions = [BasePositionR_body, BasePositionR_body+BasePositionR_panel, BasePositionL_body, BasePositionL_body+BasePositionL_panel,]

print(linkInertialFramePositions)
#Note the orientations are given in quaternions (4 params). There are function to convert of Euler angles and back
linkInertialFrameOrientations=[[0,0,0,1]]*nlnk
#indices determine for each link which other link it is attached to
# for example 3rd index = 2 means that the front left knee jjoint is attached to the front left hip
indices=[0, 1, 0, 3]
#Most joint are revolving. The prismatic joints are kept fixed for now
jointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_REVOLUTE]# JOINT_SPHERICAL,JOINT_REVOLUTE
#revolution axis for each revolving joint
axis=[[0,0,1], [0,0,1], [0,0,1], [0,0,1]]

#Drop the body in the scene at the following body coordinates
basePosition = [0,0,0]
baseOrientation = [0,0,0,1]
#Main function that creates the dog
sat = p.createMultiBody(body_Mass,Sat_body,visualShapeId,basePosition,baseOrientation,
                        linkMasses=link_Masses,
                        linkCollisionShapeIndices=linkCollisionShapeIndices,
                        linkVisualShapeIndices=linkVisualShapeIndices,
                        linkPositions=linkPositions,
                        linkOrientations=linkOrientations,
                        linkInertialFramePositions=linkInertialFramePositions,
                        linkInertialFrameOrientations=linkInertialFrameOrientations,
                        linkParentIndices=indices,
                        linkJointTypes=jointTypes,
                        linkJointAxis=axis		)#	

#Add earth like gravity
p.setGravity(0,0,0)

while(True):
    p.stepSimulation()