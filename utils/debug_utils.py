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
    q = [-3.0032, -1.4854,  0.7353, -1.0663, -0.0398,  1.5665]
    control_points = [[0.5000, 0.7000, 0.0892],
        [0.5076, 0.6455, 0.0892],
        [0.4924, 0.7545, 0.0892],
        [0.5545, 0.7076, 0.0892],
        [0.5187, 0.5654, 0.0892],
        [0.5086, 0.5640, 0.2087],
        [0.5263, 0.5110, 0.0892],
        [0.5112, 0.6199, 0.0892],
        [0.4663, 0.6790, 0.5126],
        [0.5823, 0.6952, 0.4035],
        [0.6402, 0.7032, 0.3490],
        [0.4970, 0.7237, 0.4785],
        [0.5081, 0.6444, 0.4785],
        [0.4246, 0.7136, 0.5467],
        [0.4356, 0.6343, 0.5467],
        [0.1821, 0.6394, 0.7800],
        [0.2109, 0.6434, 0.7727],
        [0.1293, 0.6320, 0.7934],
        [0.1893, 0.6404, 0.8091],
        [0.1949, 0.5473, 0.7800],
        [0.2023, 0.5471, 0.8091],
        [0.1875, 0.5475, 0.7509],
        [0.2478, 0.5547, 0.7666],
        [0.1040, 0.5346, 0.8030],
        [0.1328, 0.5386, 0.7956],
        [0.0752, 0.5306, 0.8104],
        [0.0965, 0.5348, 0.7740],
        [0.1264, 0.5882, 0.7975],
        [0.0688, 0.5802, 0.8124],
        [0.0901, 0.5843, 0.7759],
        [0.1146, 0.4531, 0.7998],
        [0.1300, 0.3341, 0.7952],
        [0.0858, 0.4491, 0.8073],
        [0.1434, 0.4571, 0.7924],
        [0.1221, 0.4529, 0.8289],
        [0.1071, 0.4532, 0.7708],
        [0.0688, 0.3912, 0.8113],
        [0.1744, 0.4059, 0.7841],
        [0.0772, 0.3268, 0.8088],
        [0.1827, 0.3415, 0.7816]]
    
    env.reset_robot(env.robot, q)
    for control_point in control_points:
        env.draw_sphere_marker(position=control_point, radius=0.02)

    print('done')

def draw_control_points_from_pcd(env, pcd):
    for control_point in pcd:
        env.draw_sphere_marker(position=control_point, radius=0.02)

    print('done')