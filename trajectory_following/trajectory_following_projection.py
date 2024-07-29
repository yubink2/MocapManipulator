import sys
import os, inspect

sys.path.append("/usr/lib/python3/dist-packages")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from scipy.spatial.transform import Rotation as R
# from scipy.optimize import minimize
# from autograd_minimize import minimize
from torchmin import minimize_constr

import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

# kinematics tools
import pytorch_kinematics as pk
import torch
import pytorch3d.transforms

# utils
from utils.transform_utils import *

class ConstraintProjection():
    def __init__(self,
                 ur5_chain,
                 T_world_to_robot_base,
                 T_world_to_human_base,
                 T_right_elbow_to_cp,
                 T_eef_to_cp,
                 device: str = 'cpu',
                 float_dtype: torch.dtype = torch.float32):
        
        # torch parameters
        self.device = device
        self.float_dtype = float_dtype

        # pytorch kinematics
        self.human_chain = pk.build_serial_chain_from_urdf(open("./urdf/humanoid_with_rev_scaled.urdf").read(), 
                                                end_link_name="right_elbow",
                                                ).to(device=self.device, dtype=self.float_dtype)
        self.ur5_chain = pk.build_serial_chain_from_urdf(open("./pybullet_ur5/urdf/ur5_robotiq_85.urdf").read(), 
                                                end_link_name="ee_link",
                                                ).to(device=self.device, dtype=self.float_dtype)
        
        # initialize transforms needed for constraint function
        self.T_world_to_robot_base = torch.Tensor(T_world_to_robot_base).to(device=self.device, dtype=self.float_dtype)
        self.T_world_to_human_base = torch.Tensor(T_world_to_human_base).to(device=self.device, dtype=self.float_dtype)
        self.T_right_elbow_to_cp = torch.Tensor(T_right_elbow_to_cp).to(device=self.device, dtype=self.float_dtype)
        self.T_eef_to_cp = torch.Tensor(T_eef_to_cp).to(device=self.device, dtype=self.float_dtype)

    def constraint_function_on_human(self, q_H, desired_cp_matrix):
        ret = self.human_chain.forward_kinematics(q_H, end_only=True)
        right_elbow_m = self.T_world_to_human_base @ ret.get_matrix().to(device=self.device, dtype=self.float_dtype)
        actual_cp_matrix = (right_elbow_m @ self.T_right_elbow_to_cp).squeeze()

        return torch.linalg.norm(actual_cp_matrix - desired_cp_matrix).to(device=self.device, dtype=self.float_dtype)
    
    def constraint_function_on_robot(self, q_R, desired_eef_matrix, q_H):
        # initialize parameters
        if not isinstance(q_R, torch.Tensor):
            q_R = torch.tensor(q_R, dtype=self.float_dtype, device=self.device)
            desired_eef_matrix = torch.tensor(desired_eef_matrix, dtype=self.float_dtype, device=self.device)
        
        if q_R.device != self.device:
            q_R = q_R.to(dtype=self.float_dtype, device=self.device)
        if desired_eef_matrix.device != self.device:
            desired_eef_matrix = desired_eef_matrix.to(dtype=self.float_dtype, device=self.device)

        # compute initial actual_eef
        link_transformation = self.ur5_chain.forward_kinematics(q_R, end_only=True)
        T_world_to_actual_eef = self.T_world_to_robot_base @ link_transformation[0].get_matrix().to(device=self.device, dtype=self.float_dtype).squeeze()
        T_world_to_actual_cp =  T_world_to_actual_eef @ self.T_eef_to_cp
        
        # constrain on human arm and recompute actual_eef
        # q_H_solution = self.constraint_manifold_on_human(T_world_to_actual_cp)
        q_H_solution, _ = self.jacobian_ik_solver(T_world_to_actual_cp, q_H)
        ret = self.human_chain.forward_kinematics(q_H_solution, end_only=True)
        T_world_to_actual_right_elbow = self.T_world_to_human_base @ ret.get_matrix().to(device=self.device, dtype=self.float_dtype)
        T_world_to_actual_cp = (T_world_to_actual_right_elbow @ self.T_right_elbow_to_cp).squeeze()
        T_world_to_actual_eef = T_world_to_actual_cp @ self.T_eef_to_cp
        actual_eef_matrix = T_world_to_actual_eef.squeeze()

        return torch.linalg.norm(actual_eef_matrix - desired_eef_matrix).to(device=self.device, dtype=self.float_dtype)
    
    def constraint_manifold_on_human(self, desired_cp_matrix):
        if desired_cp_matrix.device != self.device:
            desired_cp_matrix = desired_cp_matrix.to(dtype=self.float_dtype, device=self.device)

        # bounds for each joint angle of human arm 
        bounds = torch.tensor([(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14), (0, 3.14)]).to(device=self.device, dtype=self.float_dtype)
        lb = torch.tensor([-3.14, -3.14, -3.14, 0])
        ub = torch.tensor([3.14, 3.14, 3.14, 3.14])

        # optimization to find q_H that satisfy the constraint
        initial_guess = torch.zeros(4).to(device=self.device, dtype=self.float_dtype)
        result = minimize_constr(lambda initial_guess: self.constraint_function_on_human(initial_guess, desired_cp_matrix), initial_guess, bounds=dict(lb=lb, ub=ub))
        
        if result.success:
            # solution, its value of constraint func, jacobian at the solution point
            return result.x, result.fun, # result.jac
        else:
            raise ValueError("Optimization failed to find a valid human arm configuration")
        
    def jacobian_ik_solver(self, desired_cp_matrix, q_H, max_iterations=500, tolerance=1e-3):
        if desired_cp_matrix.device != self.device:
            desired_cp_matrix = desired_cp_matrix.to(dtype=self.float_dtype, device=self.device)

        # q_H_sol = torch.tensor(q_H, requires_grad=True, dtype=self.float_dtype, device=self.device)
        q_H_sol = torch.zeros(4, requires_grad=True, dtype=self.float_dtype, device=self.device)
        lb = torch.tensor([-3.14, -3.14, -3.14, 0], device=self.device, dtype=self.float_dtype)
        ub = torch.tensor([3.14, 3.14, 3.14, 3.14], device=self.device, dtype=self.float_dtype)
        
        optimizer = torch.optim.Adam([q_H_sol], lr=1e-1)
        
        for i in range(max_iterations):
            optimizer.zero_grad()
            loss = self.constraint_function_on_human(q_H_sol, desired_cp_matrix)
            loss.backward(retain_graph=True)
            optimizer.step()

            with torch.no_grad():
                q_H_sol.data = torch.clamp(q_H_sol.data, lb, ub)
            
            if loss.item() < tolerance:
                break

        # if loss.item() > tolerance:
        #     raise ValueError("didnt converge to a solution")
        
        print(q_H_sol, loss.item())
        return q_H_sol, loss.item()
        
    def compute_constraint_jacobian_on_robot(self, q_R, desired_eef_matrix, q_H):
        q_R = torch.tensor(q_R, requires_grad=True, device=self.device, dtype=self.float_dtype)
        desired_eef_matrix = torch.tensor(desired_eef_matrix, requires_grad=True, device=self.device, dtype=self.float_dtype)
        if q_H.device != self.device:
            q_H = q_H.to(dtype=self.float_dtype, device=self.device)

        constraint_val = self.constraint_function_on_robot(q_R, desired_eef_matrix, q_H)
        constraint_val.backward()
        jacobian = q_R.grad

        return jacobian