import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

def visualize_point_cloud(pcd):
    pc_ply = o3d.geometry.PointCloud()
    pc_ply.points = o3d.utility.Vector3dVector(pcd)
    o3d.visualization.draw_geometries([pc_ply])

# Function to generate surface vertices for a capsule shape
def generate_capsule_vertices(radius, height, position, orientation, resolution=15):
    vertices = []

    # Generate vertices for the cylindrical part
    for i in range(resolution + 1):
        angle = i * 2 * np.pi / resolution
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        for z in np.linspace(-height / 2, height / 2, resolution):
            vertices.append([x, y, z])

    # Generate vertices for the top hemispherical end
    for i in range(resolution + 1):
        for j in range(resolution // 2 + 1):
            theta = i * 2 * np.pi / resolution
            phi = j * np.pi / (resolution // 2)
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z_top = radius * np.cos(phi) + height / 2
            vertices.append([x, y, z_top])

    # Generate vertices for the bottom hemispherical end
    for i in range(resolution + 1):
        for j in range(resolution // 2 + 1):
            theta = i * 2 * np.pi / resolution
            phi = j * np.pi / (resolution // 2)
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z_bottom = -radius * np.cos(phi) - height / 2
            vertices.append([x, y, z_bottom])

    # Convert vertices to a numpy array for easier transformation
    vertices = np.array(vertices)

    # Create a rotation matrix from the orientation quaternion
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

    # Apply the rotation and translation to each vertex
    transformed_vertices = []
    for vertex in vertices:
        rotated_vertex = np.dot(rotation_matrix, vertex)
        transformed_vertex = rotated_vertex + position
        transformed_vertices.append(transformed_vertex)

    return np.array(transformed_vertices)

# Function to generate surface vertices for a sphere
def generate_sphere_vertices(radius, position, orientation, resolution=15):
    vertices = []
    for i in range(resolution + 1):
        for j in range(resolution + 1):
            theta = i * 2 * np.pi / resolution
            phi = j * np.pi / resolution
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            vertices.append([x, y, z])
    
    # Convert vertices to a numpy array for easier transformation
    vertices = np.array(vertices)
    
    # Create a rotation matrix from the orientation quaternion
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    
    # Apply the rotation and translation to each vertex
    transformed_vertices = []
    for vertex in vertices:
        rotated_vertex = np.dot(rotation_matrix, vertex)
        transformed_vertex = rotated_vertex + position
        transformed_vertices.append(transformed_vertex)
    
    return transformed_vertices

# Function to generate surface vertices for a box with position and orientation
def generate_box_vertices(half_extents, position, orientation, resolution=15):
    # Generate vertices for a box centered at the origin
    x_range = np.linspace(-half_extents[0]/2, half_extents[0]/2, resolution)
    y_range = np.linspace(-half_extents[1]/2, half_extents[1]/2, resolution)
    z_range = np.linspace(-half_extents[2]/2, half_extents[2]/2, resolution)
    vertices = []

    # Generate vertices for each face of the box
    for x in x_range:
        for y in y_range:
            vertices.append([x, y, -half_extents[2]/2])
            vertices.append([x, y, half_extents[2]/2])
    for x in x_range:
        for z in z_range:
            vertices.append([x, -half_extents[1]/2, z])
            vertices.append([x, half_extents[1]/2, z])
    for y in y_range:
        for z in z_range:
            vertices.append([-half_extents[0]/2, y, z])
            vertices.append([half_extents[0]/2, y, z])
    
    # Convert vertices to a numpy array for easier transformation
    vertices = np.array(vertices)
    
    # Create a rotation matrix from the orientation quaternion
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    
    # Apply the rotation and translation to each vertex
    transformed_vertices = []
    for vertex in vertices:
        rotated_vertex = np.dot(rotation_matrix, vertex)
        transformed_vertex = rotated_vertex + position
        transformed_vertices.append(transformed_vertex)
    
    return transformed_vertices

# Function to generate surface vertices for a cylinder
def generate_cylinder_vertices(radius, height, position, orientation, resolution=15):
    vertices = []
    
    # Generate vertices for the circular bases
    for i in range(resolution + 1):
        angle = i * 2 * np.pi / resolution
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append([x, y, -height / 2])  # Bottom circle
        vertices.append([x, y, height / 2])   # Top circle
    
    # Generate vertices for the vertical lines
    for i in range(resolution + 1):
        angle = i * 2 * np.pi / resolution
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        for j in np.linspace(-height / 2, height / 2, resolution):
            vertices.append([x, y, j])
    
    # Convert vertices to a numpy array for easier transformation
    vertices = np.array(vertices)
    
    # Create a rotation matrix from the orientation quaternion
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    
    # Apply the rotation and translation to each vertex
    transformed_vertices = []
    for vertex in vertices:
        rotated_vertex = np.dot(rotation_matrix, vertex)
        transformed_vertex = rotated_vertex + position
        transformed_vertices.append(transformed_vertex)
    
    return transformed_vertices

# Function to get point cloud from collision shape vertices
def get_point_cloud_from_collision_shapes(body_id, box_shape=None):
    point_cloud = []
    for link_id in range(-1, p.getNumJoints(body_id)):
        collision_shapes = p.getCollisionShapeData(body_id, link_id)
        for shape in collision_shapes:
            if shape[1] == link_id:
                mesh_data = p.getCollisionShapeData(body_id, link_id)
                for data in mesh_data:
                    if data[2] in [p.GEOM_MESH, p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
                        position = data[5]
                        orientation = data[6]

                        if data[2] == p.GEOM_MESH:
                            mesh_file = data[4].decode("utf-8")
                            mesh_scale = data[3]
                            vertices = p.getMeshData(body_id, link_id)[1]
                        elif data[2] == p.GEOM_BOX:
                            half_extents = np.array(data[3])
                            if box_shape is not None:
                                vertices = generate_box_vertices(box_shape, position, orientation)
                            else:
                                vertices = generate_box_vertices(half_extents, position, orientation)
                        elif data[2] == p.GEOM_SPHERE:
                            radius = data[3][0]
                            vertices = generate_sphere_vertices(radius, position, orientation)
                        elif data[2] == p.GEOM_CYLINDER:
                            height = data[3][0]
                            radius = data[3][1]
                            vertices = generate_cylinder_vertices(radius, height, position, orientation)
                        elif data[2] == p.GEOM_CAPSULE:
                            height = data[3][0]
                            radius = data[3][1]
                            vertices = generate_capsule_vertices(radius, height, position, orientation)

                        if link_id == -1: 
                            link_state = p.getBasePositionAndOrientation(body_id)
                        else:
                            link_state = p.getLinkState(body_id, link_id)

                        link_world_pos = link_state[0]
                        link_world_ori = link_state[1]
                        link_world_transform = p.getMatrixFromQuaternion(link_world_ori)
                        link_world_transform = np.array(link_world_transform).reshape(3, 3)
                        
                        for vertex in vertices:
                            vertex_world = np.dot(link_world_transform, vertex) + np.array(link_world_pos)
                            point_cloud.append(vertex_world)

                        # # visuaalize point cloud for debugging
                        # visualize_point_cloud(point_cloud)
    return point_cloud

# Function to get point cloud from collision shape vertices from one specific link
def get_point_cloud_from_collision_shapes_specific_link(body_id, link_id, resolution=15):
    point_cloud = []
    collision_shapes = p.getCollisionShapeData(body_id, link_id)
    for data in collision_shapes:
        if data[2] in [p.GEOM_MESH, p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
            position = data[5]
            orientation = data[6]

            if data[2] == p.GEOM_MESH:
                mesh_file = data[4].decode("utf-8")
                mesh_scale = data[3]
                vertices = p.getMeshData(body_id, link_id)[1]
            elif data[2] == p.GEOM_BOX:
                half_extents = np.array(data[3])
                vertices = generate_box_vertices(half_extents, position, orientation, resolution)
            elif data[2] == p.GEOM_SPHERE:
                radius = data[3][0]
                vertices = generate_sphere_vertices(radius, position, orientation, resolution)
            elif data[2] == p.GEOM_CYLINDER:
                height = data[3][0]
                radius = data[3][1]
                vertices = generate_cylinder_vertices(radius, height, position, orientation, resolution)
            elif data[2] == p.GEOM_CAPSULE:
                height = data[3][0]
                radius = data[3][1]
                vertices = generate_capsule_vertices(radius, height, position, orientation, resolution)

            if link_id == -1: 
                link_state = p.getBasePositionAndOrientation(body_id)
            else:
                link_state = p.getLinkState(body_id, link_id)

            link_world_pos = link_state[0]
            link_world_ori = link_state[1]
            link_world_transform = p.getMatrixFromQuaternion(link_world_ori)
            link_world_transform = np.array(link_world_transform).reshape(3, 3)
            
            for vertex in vertices:
                vertex_world = np.dot(link_world_transform, vertex) + np.array(link_world_pos)
                point_cloud.append(vertex_world)

    return point_cloud

# Function to get point cloud from visual shape vertices
def get_point_cloud_from_visual_shapes(body_id):
    point_cloud = []
    for link_id in range(-1, p.getNumJoints(body_id)):
        visual_shapes = p.getVisualShapeData(body_id)
        for shape in visual_shapes:
            if shape[1] == link_id:
                mesh_data = p.getVisualShapeData(body_id, link_id)
                for data in mesh_data:
                    if data[2] in [p.GEOM_MESH, p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
                        position = data[5]
                        orientation = data[6]

                        if data[2] == p.GEOM_MESH:
                            mesh_file = data[4].decode("utf-8")
                            mesh_scale = data[3]
                            vertices = p.getMeshData(body_id, link_id)[1]
                        elif data[2] == p.GEOM_BOX:
                            half_extents = np.array(data[3])
                            vertices = generate_box_vertices(half_extents, position, orientation)
                        elif data[2] == p.GEOM_SPHERE:
                            radius = data[3][0]
                            vertices = generate_sphere_vertices(radius, position, orientation)
                        elif data[2] == p.GEOM_CYLINDER:
                            height = data[3][0]
                            radius = data[3][1]
                            vertices = generate_cylinder_vertices(radius, height, position, orientation)
                        elif data[2] == p.GEOM_CAPSULE:
                            height = data[3][0]
                            radius = data[3][1]
                            vertices = generate_capsule_vertices(radius, height, position, orientation)

                        if link_id == -1: 
                            link_state = p.getBasePositionAndOrientation(body_id)
                        else:
                            link_state = p.getLinkState(body_id, link_id)

                        link_world_pos = link_state[0]
                        link_world_ori = link_state[1]
                        link_world_transform = p.getMatrixFromQuaternion(link_world_ori)
                        link_world_transform = np.array(link_world_transform).reshape(3, 3)
                        
                        for vertex in vertices:
                            vertex_world = np.dot(link_world_transform, vertex) + np.array(link_world_pos)
                            point_cloud.append(vertex_world)

                        # # visuaalize point cloud for debugging
                        # visualize_point_cloud(point_cloud)
    return point_cloud

# Function to get point cloud from humanoid specifying which link to separate
def get_humanoid_point_cloud(body_id, link_id_to_separate=None):
    separate_point_cloud = []
    if link_id_to_separate is not None:
        for link_id in link_id_to_separate:
            mesh_data = p.getCollisionShapeData(body_id, link_id)
            for data in mesh_data:
                if data[2] in [p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CAPSULE, p.GEOM_CYLINDER, p.GEOM_MESH]:
                    position = data[5]
                    orientation = data[6]
                    
                    if data[2] == p.GEOM_BOX:
                        half_extents = np.array(data[3])
                        vertices = generate_box_vertices(half_extents, position, orientation, resolution=8)
                    elif data[2] == p.GEOM_SPHERE:
                        radius = data[3][0]
                        vertices = generate_sphere_vertices(radius, position, orientation, resolution=8)
                    elif data[2] == p.GEOM_CAPSULE:
                        height = data[3][0]
                        radius = data[3][1]
                        vertices = generate_capsule_vertices(radius, height, position, orientation, resolution=8)
                    elif data[2] == p.GEOM_CYLINDER:
                        height = data[3][0]
                        radius = data[3][1]
                        vertices = generate_cylinder_vertices(radius, height, position, orientation, resolution=8)
                    elif data[2] == p.GEOM_MESH:
                        mesh_file = data[4].decode("utf-8")
                        mesh_scale = data[3]
                        vertices = p.getMeshData(body_id, link_id)[1]

                    if link_id == -1: 
                        link_state = p.getBasePositionAndOrientation(body_id)
                    else:
                        link_state = p.getLinkState(body_id, link_id)

                    link_world_pos = link_state[0]
                    link_world_ori = link_state[1]
                    link_world_transform = p.getMatrixFromQuaternion(link_world_ori)
                    link_world_transform = np.array(link_world_transform).reshape(3, 3)
                    
                    for vertex in vertices:
                        vertex_world = np.dot(link_world_transform, vertex) + np.array(link_world_pos)
                        separate_point_cloud.append(vertex_world)

    point_cloud = []
    for link_id in range(-1, p.getNumJoints(body_id)):
        # skip link_id_to_separate
        if link_id_to_separate is not None:
            if link_id in link_id_to_separate:
                continue
        
        # skip link = right shoulder
        if link_id == 6:
            continue

        # generate point cloud
        collision_shapes = p.getCollisionShapeData(body_id, link_id)
        for shape in collision_shapes:
            if shape[1] == link_id:
                mesh_data = p.getCollisionShapeData(body_id, link_id)
                for data in mesh_data:
                    if data[2] in [p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CAPSULE, p.GEOM_CYLINDER]:
                        position = data[5]
                        orientation = data[6]
                
                        if data[2] == p.GEOM_BOX:
                            half_extents = np.array(data[3])
                            vertices = generate_box_vertices(half_extents, position, orientation, resolution=8)
                        elif data[2] == p.GEOM_SPHERE:
                            radius = data[3][0]
                            vertices = generate_sphere_vertices(radius, position, orientation, resolution=8)
                        elif data[2] == p.GEOM_CAPSULE:
                            height = data[3][0]
                            radius = data[3][1]
                            vertices = generate_capsule_vertices(radius, height, position, orientation, resolution=8)
                        elif data[2] == p.GEOM_CYLINDER:
                            height = data[3][0]
                            radius = data[3][1]
                            vertices = generate_cylinder_vertices(radius, height, position, orientation, resolution=8)

                        if link_id == -1: 
                            link_state = p.getBasePositionAndOrientation(body_id)
                        else:
                            link_state = p.getLinkState(body_id, link_id)

                        link_world_pos = link_state[0]
                        link_world_ori = link_state[1]
                        link_world_transform = p.getMatrixFromQuaternion(link_world_ori)
                        link_world_transform = np.array(link_world_transform).reshape(3, 3)
                        
                        for vertex in vertices:
                            vertex_world = np.dot(link_world_transform, vertex) + np.array(link_world_pos)
                            point_cloud.append(vertex_world)
    
    return point_cloud, separate_point_cloud

if __name__ == "__main__":
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load the plane and a few objects
    p.loadURDF("plane.urdf")
    # r2d2 = p.loadURDF("r2d2.urdf", [1, 1, 0.5])
    # duck = p.loadURDF("duck_vhacd.urdf", [1, -1, 0.5])
    # capsule = p.loadURDF("urdf/capsule_0.urdf", [0,0,0])
    humanoid = p.loadURDF("humanoid/humanoid.urdf", [0,0,0])

    # Get point clouds from all objects
    point_cloud = []
    point_cloud.extend(get_humanoid_point_cloud(humanoid)[0])

    # Convert to numpy array
    point_cloud = np.array(point_cloud)

    # Visualize the point cloud
    pc_ply = o3d.geometry.PointCloud()
    pc_ply.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pc_ply])
    
    # Disconnect from PyBullet
    p.disconnect()