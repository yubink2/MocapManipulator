import sys
import os, inspect

sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import numpy as np
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R
import time

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

# kinematics tools
import pytorch_kinematics as pk
import torch

# minimize
# from scipy.optimize import minimize
from torchmin import minimize_constr

# utils
from utils.transform_utils import *


# load environment
bc = BulletClient(connection_mode=p.GUI)
bc.setAdditionalSearchPath(pybullet_data.getDataPath())
bc.setGravity(0, 0, -9.8)

humanoid = bc.loadURDF("./urdf/humanoid_with_rev_scaled.urdf",
                        useFixedBase=True)

right_shoulder_y = 3
right_shoulder_p = 4
right_shoulder_r = 5
right_shoulder = 6
right_elbow = 7
right_wrist = 8

# pytorch kinematics
chain = pk.build_serial_chain_from_urdf(open("./urdf/humanoid_with_rev_scaled.urdf").read(), 
                                        end_link_name="right_elbow",
                                        ).to(device="cpu", dtype=torch.float64)
humanoid_base = bc.getBasePositionAndOrientation(humanoid)[:2]
T_world_to_base = torch.Tensor(compute_matrix(translation=humanoid_base[0], rotation=humanoid_base[1]), device="cpu").to(dtype=torch.float64)

# transforms
cp_pose = bc.getLinkState(humanoid, right_elbow)[:2]
right_elbow_pose = bc.getLinkState(humanoid, right_elbow)[4:6]
right_elbow_pose_inverse = bc.invertTransform(right_elbow_pose[0], right_elbow_pose[1])
right_elbow_to_cp = bc.multiplyTransforms(right_elbow_pose_inverse[0], right_elbow_pose_inverse[1],
                                            cp_pose[0], cp_pose[1])
T_right_elbow_to_cp = torch.tensor(compute_matrix(translation=right_elbow_to_cp[0], rotation=right_elbow_to_cp[1]))

def draw_frame(position, quaternion=[0, 0, 0, 1]):
    m = R.from_quat(quaternion).as_matrix()
    x_vec = m[:, 0]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for color, column in zip(colors, range(3)):
        vec = m[:, column]
        from_p = position
        to_p = position + (vec * 0.1)
        bc.addUserDebugLine(from_p, to_p, color, lineWidth=3, lifeTime=0)

def reset_human_arm(q_human):
    bc.resetJointState(humanoid, right_shoulder_y, q_human[0])
    bc.resetJointState(humanoid, right_shoulder_p, q_human[1])
    bc.resetJointState(humanoid, right_shoulder_r, q_human[2])
    bc.resetJointState(humanoid, right_elbow, q_human[3])

# Define the constraint function g(q)
def constraint_function(q, desired_cp_matrix):
    # #### FORWARD KINEMATICS
    # for _ in range(10):
    #     reset_human_arm(q)
    #     bc.stepSimulation()
    # actual_position_pb = np.array(bc.getLinkState(humanoid, right_elbow)[0])

    ret = chain.forward_kinematics(q, end_only=True)
    right_elbow_m = T_world_to_base @ ret.get_matrix()
    actual_cp_matrix = (right_elbow_m @ T_right_elbow_to_cp).squeeze()

    return torch.linalg.norm(actual_cp_matrix - desired_cp_matrix) 

# Define the constraint manifold X
def constraint_manifold(desired_cp_matrix):
    
    # Initial guess for the joint angles
    initial_guess = torch.zeros(4).to(dtype=torch.float64)
    
    # Define the bounds for each joint angle
    bounds = torch.tensor([(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 3.14)])
    lb = torch.tensor([-3.14, -3.14, -3.14, 0])
    ub = torch.tensor([3.14, 3.14, 3.14, 3.14])
    
    # Perform the optimization to find the joint angles that satisfy the constraint
    # result = minimize(constraint_function, initial_guess, args=(desired_cp_matrix,T_right_elbow_to_cp,), bounds=bounds)
    result = minimize_constr(lambda initial_guess: constraint_function(initial_guess, desired_cp_matrix), initial_guess, bounds=dict(lb=lb, ub=ub))
    
    if result.success:
        # solution, its value of constraint func, jacobian at the solution point
        return result.x, result.fun, result.jac
    else:
        raise ValueError("Optimization failed to find a valid configuration.")

def compute_constraint_jacobian(q, desired_cp_matrix):
    q_tensor = torch.tensor(q, requires_grad=True).to(dtype=torch.float64)
    constraint_val = constraint_function(q_tensor, desired_cp_matrix)
    constraint_val.backward()
    jacobian = q_tensor.grad
    return jacobian

def jacobian_ik_solver(desired_cp_matrix, max_iterations=100, tolerance=1e-3):
    q_H = torch.zeros(4, requires_grad=True, dtype=torch.float64, device="cpu")
    lb = torch.tensor([-3.14, -3.14, -3.14, 0], device="cpu", dtype=torch.float64)
    ub = torch.tensor([3.14, 3.14, 3.14, 3.14], device="cpu", dtype=torch.float64)
    
    optimizer = torch.optim.Adam([q_H], lr=1e-1)
    
    for i in range(max_iterations):
        optimizer.zero_grad()
        loss = constraint_function(q_H, desired_cp_matrix)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            q_H.data = torch.clamp(q_H.data, lb, ub)
        
        if loss.item() < tolerance:
            break
    
    return q_H, loss.item()

if __name__ == "__main__":
    ### valid
    # target_position = np.array([cp_pose[0][0]+0.2, cp_pose[0][1]+0.5, cp_pose[0][2]])

    # q_H = bc.calculateInverseKinematics(humanoid, right_wrist, targetPosition=target_position)
    # reset_human_arm(q_H)
    # bc.stepSimulation()
    # desired_cp_pose = bc.getLinkState(humanoid, right_elbow)[:2]

    # for _ in range(100):
    #     bc.stepSimulation()

    desired_cp_pose = ((0.24307311568325185, 0.5164632176548571, 0.1611376), (0.0, 0.0, 0.994199387039995, 0.10755267922882439))

    desired_matrix_tensor = torch.tensor(compute_matrix(translation=desired_cp_pose[0], rotation=desired_cp_pose[1]))
    
    # q_solution, q_solution_value, q_solution_jac = constraint_manifold(desired_matrix_tensor)
    # print("Joint angles that satisfy the constraint:", q_solution, q_solution_value, q_solution_jac)

    prev_time = time.time()
    q_solution, dist = jacobian_ik_solver(desired_matrix_tensor)
    print('jacobian ik: ', time.time()-prev_time)
    print(q_solution)
    print(dist)

    prev_time = time.time()
    q_H = bc.calculateInverseKinematics(humanoid, right_wrist, targetPosition=desired_cp_pose[0], targetOrientation=desired_cp_pose[1])
    print('pybullet ik: ', time.time()-prev_time)
    print(q_H)
    
    draw_frame(desired_cp_pose[0])
    for _ in range(10):
        reset_human_arm(q_solution)
        bc.stepSimulation()
    print('done')


    # print(q_solution_jac)
    # print(compute_constraint_jacobian(q_solution, desired_matrix_tensor))

    # # invalid
    # desired_position = np.array([cp_pose[0][0]-0.3, cp_pose[0][1]-0.3, cp_pose[0][2]])
    # q_solution, _, _ = constraint_manifold(desired_position)
    # print("Joint angles that satisfy the constraint:", q_solution)
    # draw_frame(desired_position)
    # for _ in range(10):
    #     reset_human_arm(q_solution)
    #     bc.stepSimulation()

    # print('done')