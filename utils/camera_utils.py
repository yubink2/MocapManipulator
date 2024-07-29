import numpy as np
import pybullet as p

### CAMERA IMAGE
def get_camera_img(env):
    ### Desired image size and intrinsic camera parameters
    ### Intrinsic camera matrix from 4th Oct 2021:    543.820     0.283204    314.424;
    ###                                               0.00000     546.691     237.466;
    width = 480
    height = 480
    f_x = 543.820
    f_y = 546.691
    c_x = 314.424
    c_y = 237.466
    skew = 0.283204

    ### The far and near values depend on the min,max desired distance from the object
    far = 1.5
    near = 0.1

    ### Camera intrinsic:
    opengl_projection_matrix = (f_x/width,            0,                      0,                      0,
                                skew/width,           f_y/height,           0,                      0,
                                2*(c_x+0.5)/width-1,  2*(c_y+0.5)/height-1,   -(far+near)/(far-near), -1,
                                0,                    0,                      -2*far*near/(far-near), 0)

    obj_pb_id = env.humanoid._humanoid
    obj_link_id = env.right_elbow

    aabb = p.getAABB(obj_pb_id, obj_link_id, physicsClientId=env.bc._client)
    aabb_min = np.array(aabb[0])
    aabb_max = np.array(aabb[1])
    obj_center = list((aabb_max + aabb_min)/2)

    camera_look_at = obj_center

    phi_deg = 130
    theta_deg = 15
    camera_distance = 0.6

    phi = np.deg2rad(phi_deg)
    theta = np.deg2rad(theta_deg)
    camera_eye_position = []
    camera_eye_position.append(camera_distance*np.cos(phi)*np.sin(theta) + obj_center[0])
    camera_eye_position.append(camera_distance*np.sin(phi)*np.sin(theta) + obj_center[1])
    camera_eye_position.append(camera_distance*np.cos(theta) + obj_center[2])

    # camera_eye_position.append(obj_center[0])
    # camera_eye_position.append(obj_center[1])
    # camera_eye_position.append(1.0 + obj_center[2])

    view_matrix = env.bc.computeViewMatrix(
        cameraEyePosition= camera_eye_position,
        cameraTargetPosition=camera_look_at,
        cameraUpVector = [0,0,1],
        physicsClientId=env.bc._client
    )

    # T_world_to_camera = np.array(view_matrix).reshape(4,4)
    # T_world_to_camera = T_world_to_camera.T
    # world_to_camera_pos = -T_world_to_camera[:3, 3]
    # world_to_camera_orn = quaternion_from_matrix(T_world_to_camera)
    # draw_frame(env, position=world_to_camera_pos, quaternion=world_to_camera_orn)
    # T_world_to_camera = compute_matrix(translation=world_to_camera_pos, rotation=world_to_camera_orn, rotation_type='quaternion')

    view_mtx = np.array(view_matrix).reshape((4,4),order='F')
    cam_pos = np.dot(view_mtx[:3,:3].T, -view_mtx[:3,3])


    ### Get depth values using Tiny renderer
    images = env.bc.getCameraImage(height=height,
            width=width,
            viewMatrix=view_matrix,
            projectionMatrix=opengl_projection_matrix,
            shadow=True,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=env.bc._client
            )
    
    rgb = images[2][:, :, :-1]
    seg = np.reshape(images[4], [height,width])
    depth_buffer = np.reshape(images[3], [height,width]) 

    # Define the camera intrinsic matrix
    K = np.array([
        [f_x, skew, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    dict = {'rgb': rgb, 'depth': depth_buffer, 'K': K, 'seg': seg}
    np.save("img_human_scene.npy", dict)

    return T_world_to_camera