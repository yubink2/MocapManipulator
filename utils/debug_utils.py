from scipy.spatial.transform import Rotation as R
import pybullet as p

def ur5_debug_parameter(env):
        for _ in range(50):
            env.move_human_arm([3.1, 1.57, -1.57, 0])
            env.bc.stepSimulation()

        pos = env.bc.getLinkState(env.humanoid._humanoid, env.right_elbow)[0]
        pos_up = (pos[0], pos[1]+0.2, pos[2])
        print(pos)

        position_control_group = []
        position_control_group.append(env.bc.addUserDebugParameter('x', -1.5, 1.5, pos_up[0]))
        position_control_group.append(env.bc.addUserDebugParameter('y', -1.5, 1.5, pos_up[1]))
        position_control_group.append(env.bc.addUserDebugParameter('z', -1.5, 1.5, pos_up[2]))
        position_control_group.append(env.bc.addUserDebugParameter('roll', -3.14, 3.14, 3.14))
        position_control_group.append(env.bc.addUserDebugParameter('pitch', -3.14, 3.14, 0))
        position_control_group.append(env.bc.addUserDebugParameter('yaw', -3.14, 3.14, -1.57))
        position_control_group.append(env.bc.addUserDebugParameter('gripper_opening', 0, 0.085, 0.08))

        while True:
            env.bc.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 

            parameter = []
            for i in range(len(position_control_group)):
                parameter.append(env.bc.readUserDebugParameter(position_control_group[i]))

            env.robot.move_ee(action=parameter[:-1], control_method='end')
            env.robot.move_gripper(parameter[-1])

            # env.move_human_arm([2.7, 1.4, -1.9, 0])

            env.bc.stepSimulation()

            # for i, joint_id in enumerate(env.robot.arm_controllable_joints):
            #     print(i, env.bc.getJointState(env.robot.id, joint_id)[0])

            # right_arm = []
            # right_arm.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_y)[0])
            # right_arm.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_p)[0])
            # right_arm.append(env.bc.getJointState(env.humanoid._humanoid, env.right_shoulder_r)[0])
            # right_arm.append(env.bc.getJointState(env.humanoid._humanoid, env.right_elbow)[0])
            # print(right_arm)


def draw_frame(env, position, quaternion=[0, 0, 0, 1]):
        m = R.from_quat(quaternion).as_matrix()
        x_vec = m[:, 0]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for color, column in zip(colors, range(3)):
            vec = m[:, column]
            from_p = position
            to_p = position + (vec * 0.1)
            env.bc.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)

def draw_control_points(env):
    q = [-2.9565, -1.9752,  1.3886, -1.1024, -0.2017,  1.3261]
    control_points = [[0.5101, 0.6459, 0.0892],
        [0.4899, 0.7541, 0.0892],
        [0.5000, 0.7000, 0.1442],
        [0.5000, 0.7000, 0.0342],
        [0.5351, 0.5124, 0.0892],
        [0.5149, 0.6205, 0.0892],
        [0.6972, 0.7815, 0.4522],
        [0.7193, 0.6636, 0.4522],
        [0.6154, 0.7662, 0.5076],
        [0.6375, 0.6483, 0.5076],
        [0.3381, 0.6533, 0.6275],
        [0.3544, 0.6563, 0.7665],
        [0.2779, 0.6420, 0.7053],
        [0.3048, 0.5524, 0.7041],
        [0.4219, 0.5743, 0.6899],
        [0.3528, 0.5757, 0.6289],
        [0.2939, 0.5385, 0.7631],
        [0.2481, 0.5536, 0.6533],
        [0.2164, 0.5329, 0.7292],
        [0.2934, 0.4084, 0.6799],
        [0.2745, 0.5245, 0.7037],
        [0.2610, 0.4740, 0.6369],
        [0.3069, 0.4589, 0.7467],
        [0.2063, 0.4608, 0.6579],
        [0.2522, 0.4457, 0.7677],
        [0.3116, 0.3711, 0.7017],
        [0.2886, 0.3828, 0.6482],
        [0.2656, 0.3944, 0.5947],
        [0.3469, 0.3279, 0.6771],
        [0.3239, 0.3395, 0.6236],
        [0.3009, 0.3511, 0.5701],
        [0.2914, 0.3138, 0.6980],
        [0.2684, 0.3254, 0.6445],
        [0.2453, 0.3370, 0.5910],
        [0.3116, 0.3711, 0.7017],
        [0.2886, 0.3828, 0.6482],
        [0.2656, 0.3944, 0.5947],
        [0.2569, 0.3677, 0.5536],
        [0.2842, 0.3540, 0.6170],
        [0.2569, 0.3677, 0.5536],
        [0.2842, 0.3540, 0.6170],
        [0.2569, 0.3677, 0.5536],
        [0.2842, 0.3540, 0.6170],
        [0.2569, 0.3677, 0.5536],
        [0.2842, 0.3540, 0.6170],
        [0.3303, 0.3307, 0.7240],
        [0.3030, 0.3445, 0.6606],
        [0.3303, 0.3307, 0.7240],
        [0.3030, 0.3445, 0.6606],
        [0.3303, 0.3307, 0.7240],
        [0.3030, 0.3445, 0.6606],
        [0.3303, 0.3307, 0.7240],
        [0.3030, 0.3445, 0.6606]]
    
    env.reset_robot(env.robot, q)
    for control_point in control_points:
        env.draw_sphere_marker(position=control_point, radius=0.03)

    print('done')

def draw_control_points_from_pcd(env, pcd):
    for control_point in pcd:
        env.draw_sphere_marker(position=control_point, radius=0.03)

    print('done')