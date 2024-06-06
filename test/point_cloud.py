import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane and a few objects
p.loadURDF("plane.urdf")
duck = p.loadURDF("duck_vhacd.urdf", [1, 0, 0.5])
cube = p.loadURDF("cube.urdf", [0, 0, 0.5])
sphere = p.loadURDF("sphere2.urdf", [-1, 0, 0.5])


# Function to generate vertices for a capsule shape
def generate_capsule_vertices(radius, height, resolution=30):
    vertices = []
    # Cylinder part
    for i in range(resolution + 1):
        angle = i * 2 * np.pi / resolution
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        for j in np.linspace(-height / 2, height / 2, resolution):
            vertices.append([x, y, j])
    # Hemispherical ends
    for i in range(resolution + 1):
        for j in range(resolution // 2 + 1):
            theta = i * 2 * np.pi / resolution
            phi = j * np.pi / (resolution // 2)
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z_top = radius * np.cos(phi) + height / 2
            z_bottom = -radius * np.cos(phi) - height / 2
            vertices.append([x, y, z_top])
            vertices.append([x, y, z_bottom])
    return vertices

# Function to generate vertices for a sphere
def generate_sphere_vertices(radius, resolution=30):
    vertices = []
    for i in range(resolution + 1):
        for j in range(resolution + 1):
            theta = i * 2 * np.pi / resolution
            phi = j * np.pi / resolution
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            vertices.append([x, y, z])
    return vertices

# Function to generate vertices for a box
def generate_box_vertices(half_extents, resolution=10):
    x_range = np.linspace(-half_extents[0], half_extents[0], resolution)
    y_range = np.linspace(-half_extents[1], half_extents[1], resolution)
    z_range = np.linspace(-half_extents[2], half_extents[2], resolution)
    vertices = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                vertices.append([x, y, z])
    return vertices

# Function to generate vertices for a cylinder
def generate_cylinder_vertices(radius, height, resolution=30):
    vertices = []
    for i in range(resolution + 1):
        angle = i * 2 * np.pi / resolution
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        for j in np.linspace(-height / 2, height / 2, resolution):
            vertices.append([x, y, j])
    return vertices

# Function to get point cloud from collision shape vertices
def get_point_cloud_from_collision_shapes(body_id):
    point_cloud = []
    for link_id in range(-1, p.getNumJoints(body_id)):
        collision_shapes = p.getVisualShapeData(body_id)
        for shape in collision_shapes:
            if shape[1] == link_id:
                # mesh_data = p.getCollisionShapeData(body_id, link_id)
                mesh_data = p.getVisualShapeData(body_id, link_id)
                for data in mesh_data:
                    if data[2] in [p.GEOM_MESH, p.GEOM_BOX, p.GEOM_SPHERE, p.GEOM_CYLINDER, p.GEOM_CAPSULE]:
                        if data[2] == p.GEOM_MESH:
                            mesh_file = data[4].decode("utf-8")
                            mesh_scale = data[3]
                            vertices = p.getMeshData(body_id, link_id)[1]
                        elif data[2] == p.GEOM_BOX:
                            half_extents = np.array(data[3])
                            vertices = generate_box_vertices(half_extents)
                        elif data[2] == p.GEOM_SPHERE:
                            radius = data[3][0]
                            vertices = generate_sphere_vertices(radius)
                        elif data[2] == p.GEOM_CYLINDER:
                            radius = data[3][0]
                            height = data[3][1]
                            vertices = generate_cylinder_vertices(radius, height)
                        elif data[2] == p.GEOM_CAPSULE:
                            radius = data[3][0]
                            height = data[3][1]
                            vertices = generate_capsule_vertices(radius, height)

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

# Get point clouds from all objects
point_cloud = []
point_cloud.extend(get_point_cloud_from_collision_shapes(duck))
# point_cloud.extend(get_point_cloud_from_collision_shapes(cube))
# point_cloud.extend(get_point_cloud_from_collision_shapes(sphere))

# Convert to numpy array
point_cloud = np.array(point_cloud)
print('print point cloud')
print(point_cloud)

# # Visualize the point cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=point_cloud[:, 2], cmap='viridis', marker='.')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

pc_ply = o3d.geometry.PointCloud()
pc_ply.points = o3d.utility.Vector3dVector(point_cloud)
o3d.visualization.draw_geometries([pc_ply])

# Disconnect from PyBullet
p.disconnect()