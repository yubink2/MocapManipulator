# https://github.com/bulletphysics/bullet3/discussions/4040

import os
import numpy as np
import time
import open3d as o3d

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

os.sys.path.append("/home/exx/Yubin/deep_mimic")
os.sys.path.append("/home/exx/Yubin/deep_mimic/mocap")

# robot
import pybullet_data
from pybullet_ur5.robot import UR5Robotiq85

# humanoid
from env.motion_capture_data import MotionCaptureData
from humanoid_with_rev import Humanoid
from humanoid_with_rev import HumanoidPose


### Desired image size and intrinsic camera parameters
### Intrinsic camera matrix from 4th Oct 2021:    543.820     0.283204    314.424;
###                                               0.00000     546.691     237.466;
width = 640
height = 480
f_x = 543.820
f_y = 546.691
c_x = 314.424
c_y = 237.466
skew = 0.283204

### The far and near values depend on the min,max desired distance from the object
far = 5
near = 0.1

### Camera intrinsic:
opengl_projection_matrix = (f_x/width,            0,                      0,                      0,
                            skew/width,           f_y/height,           0,                      0,
                            2*(c_x+0.5)/width-1,  2*(c_y+0.5)/height-1,   -(far+near)/(far-near), -1,
                            0,                    0,                      -2*far*near/(far-near), 0)


# load bc server
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.configureDebugVisualizer(bc.COV_ENABLE_Y_AXIS_UP, 1)
bc.setGravity(0, -9.8, 0) 

# load environment
y2zOrn = bc.getQuaternionFromEuler((-1.57, 0, 0))
plane_id = bc.loadURDF("plane.urdf", (0, -0.04, 0), y2zOrn)
bed_id = bc.loadURDF("./urdf/bed_0.urdf", (0.0, 0.0, 0.0), y2zOrn, useFixedBase=True, globalScaling=1.2)  # bed
table1_id = bc.loadURDF("table/table.urdf", (-1.5, 0.0, 1.3), y2zOrn, globalScaling=0.6)  # table
table2_id = bc.loadURDF("table/table.urdf", (1.5, 0.0, 1.3), y2zOrn, globalScaling=0.6)  # table

# load human
motionPath = 'data/Sitting1.json'
motion = MotionCaptureData()
motion.Load(motionPath)
humanoid = Humanoid(bc, motion, [0, 0.3, 0])

right_shoulder_y = 3
right_shoulder_p = 4
right_shoulder_r = 5
right_elbow = 7

# load robot
robot = UR5Robotiq85(bc, (-0.75, 0, 0), (-1.57, 0, 0), globalScaling=1.2)
robot.load()
robot.reset()

obj_pb_id = humanoid._humanoid
print(bc._client)

obj_initial_orientation_euler = np.random.uniform(low=0.,high=360.,size=3)
scale = np.random.uniform(low=1.,high=3.)

aabb = p.getAABB(obj_pb_id,physicsClientId=bc._client)
aabb_min = np.array(aabb[0])
aabb_max = np.array(aabb[1])
obj_center = list((aabb_max + aabb_min)/2)

### Initialize point cloud and its normals
pc = []
normals = []

phis = [1, 45, 90, 135, 179]
thetas = [1, 10, 45]

### Camera position and orientation
camera_look_at = obj_center

for phi_deg in phis:
    for theta_deg in thetas:
        print(f'loop phi {phi_deg}, theta {theta_deg}')

        # camera_distance = np.random.uniform(0.35,0.55)
        camera_distance = 3

        phi = np.deg2rad(phi_deg)
        theta = np.deg2rad(theta_deg)
        camera_eye_position = []
        camera_eye_position.append(camera_distance*np.cos(phi)*np.sin(theta) + obj_center[0])
        camera_eye_position.append(camera_distance*np.sin(phi)*np.sin(theta) + obj_center[1])
        camera_eye_position.append(camera_distance*np.cos(theta) + obj_center[2])

        view_matrix = p.computeViewMatrix(
            cameraEyePosition= camera_eye_position,
            cameraTargetPosition=camera_look_at,
            cameraUpVector = [0,1,0],
            physicsClientId=bc._client
        )

        ### Get depth values using Tiny renderer
        images = p.getCameraImage(height=height,
                width=width,
                viewMatrix=view_matrix,
                projectionMatrix=opengl_projection_matrix,
                shadow=True,
                renderer=p.ER_TINY_RENDERER,
                physicsClientId=bc._client
                )
        
        rgb = np.reshape(images[2], (height,width, 4)) # RGB+alpha
        seg = np.reshape(images[4], [height,width])
        depth_buffer = np.reshape(images[3], [height,width]) 
        depth = far * near / (far - (far - near) * depth_buffer) # What is done here?
        obj_segmentation = seg > 0
        depth_buffer_seg = depth_buffer * obj_segmentation
        depth_seg = depth*obj_segmentation

        opengl_projection_matrix_np = np.transpose(np.reshape(np.array(opengl_projection_matrix), (4,4)))
        view_matrix_inv_np = np.transpose(np.reshape(np.array(view_matrix), (4,4))) # From world to camera, needs to be transposed because OpenGL is column-based
        view_matrix_np = np.linalg.inv(view_matrix_inv_np) # From camera to world

        pc_view = []            

        ### Generate point cloud
        for u in range(height):
            for v in range(width):
                if obj_segmentation[u,v]:
                    point_pixel = np.array([(2*v/width-1),(-2*u/height+1),2*depth_buffer[u,v]-1,1])
                    point_camera = np.matmul(np.linalg.inv(opengl_projection_matrix_np),point_pixel)
                    point_world = np.matmul(view_matrix_np,point_camera)
                    point_world /= point_world[3]
                    pc_view.append(point_world[:3])
                    pc.append(point_world[:3])

        pc_ply = o3d.geometry.PointCloud()
        pc_ply.points = o3d.utility.Vector3dVector(pc_view)

        pc_ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=300))        
        print("Print the normal vectors of the first 10 points")
        print(np.asarray(pc_ply.normals)[:10, :])

        pc_ply.orient_normals_towards_camera_location(camera_eye_position)
        normal = np.reshape(np.array(pc_ply.normals),np.shape(pc_view))
        # o3d.visualization.draw_geometries([pc_ply])
        
        pc_ply.clear()
        normal = normal.tolist()
        normals.extend(normal)

### Point cloud with normals per point
pc_normals = np.concatenate((pc, normals),axis=1)
dict = {'xyz': pc_normals}

np.save("pc_human_scene.npy", dict)

pc_ply = o3d.geometry.PointCloud()
pc_ply.points = o3d.utility.Vector3dVector(pc)
pc_ply.normals = o3d.utility.Vector3dVector(normals)
o3d.io.write_point_cloud("pc_human_scene.pcd", pc_ply)

cloud = o3d.io.read_point_cloud("pc_human_scene.pcd")
o3d.visualization.draw_geometries([cloud])