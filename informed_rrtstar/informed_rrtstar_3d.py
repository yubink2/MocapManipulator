import sys
import pathlib
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, currentdir)
os.sys.path.insert(0, parentdir)
# os.sys.path.insert(0, parentparentdir)
print(sys.path)

import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np

from utils.angle import rot_mat_2d
from utils.collision_utils import get_collision_fn
from utils.sample_utils import InformedSampler

import pybullet as p
import pybullet_data
import time


class InformedRRTStar:

    def __init__(self, start_config, goal_config, obstacle_list, 
                 ur5, UR5_JOINT_INDICES,
                 expand_dis=0.8, goal_sample_rate=10, max_iter=50):
        self.start = Node(start_config)
        self.goal = Node(goal_config)
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = None
        self.obstacle_list = obstacle_list

        self.dof = len(UR5_JOINT_INDICES)

        self.goal_found = False
        self.solution_set = set()

        self.collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacle_list,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())
        self.informed_sampler = InformedSampler(np.array(goal_config), np.array(start_config))

    def initialize_cbest_and_tree(self, path):
        # initialize exploration tree & c_best with initial path
        for q_idx, q in enumerate(path):
            q_node = Node(q)
            if q_idx == 0:
                self.node_list = [self.start]
            else:
                parent_idx = len(self.node_list) - 1
                q_node.cost = self.node_list[parent_idx].cost + self.distance(self.node_list[parent_idx], q_node)
                q_node.set_parent(parent_idx)
                self.node_list[parent_idx].add_child(q_idx)
                self.node_list.append(q_node)

        # re-initialize informed sampling space
        self.informed_sampler = InformedSampler(np.array(path[-1]), np.array(path[0])) 
        
        # add initial path to solution set if near goal
        last_node = self.node_list[-1]
        last_node_idx = len(self.node_list)-1
        if self.is_near_goal(last_node):
            self.solution_set.add(last_node_idx)
            self.goal_found = True
            print('init solution_set: ', self.solution_set)
            print(f'init traj: dist(last node, goal): {self.distance(last_node, self.goal)}')

        c_best = self.node_list[-1].cost
        return c_best

    def plan(self, path):
        c_best = self.initialize_cbest_and_tree(path)
        # for i, node in enumerate(self.node_list):
        #     print(f'{i} node: {node.config}, cost: {node.cost}, parent: {node.parent}, children: {node.children}')
        print('c_best: ', c_best)
        # sys.exit()

        rand_config = self.informed_sampler.sample(c_best)
        rand_node = Node(rand_config)

        for i in range(self.max_iter):
            # print(i)
            rand_config = self.informed_sampler.sample(c_best)
            rand_node = Node(rand_config)
            nearest_node_idx, nearest_node = self.find_nearest(rand_node, self.node_list)            

            # if no collision
            if nearest_node is not None and self.steer_to(rand_node, nearest_node):
                new_node = copy.deepcopy(rand_node)
                new_node.set_parent(nearest_node_idx)
                new_node.cost = nearest_node.cost + self.distance(nearest_node, new_node)

                near_inds = self.find_near_nodes(new_node)
                new_parent_idx = self.choose_parent(new_node, near_inds)
                # print('near indicies: ', near_inds)
                # print('new parent idx: ', new_parent_idx)

                if new_parent_idx is not None:
                    new_node.set_parent(new_parent_idx)
                    new_node.cost = self.node_list[new_parent_idx].cost + self.distance(self.node_list[new_parent_idx], new_node)

                self.node_list.append(new_node)
                new_node_idx = len(self.node_list) - 1
                self.node_list[new_node.parent].add_child(new_node_idx)

                # print(f'new node {new_node_idx} added, its parent: {new_node.parent}, cost: {new_node.cost}')

                self.rewire(new_node, new_node_idx, near_inds)

                # print(f'new node {new_node_idx} after rewire, its parent: {new_node.parent}, cost: {new_node.cost}')

                # if goal found
                if self.is_near_goal(new_node):
                    self.solution_set.add(new_node_idx)
                    self.goal_found = True
                    print('solution_set: ', self.solution_set)
                    print(f'dist(new node, goal): {self.distance(new_node, self.goal)}')

        # for i, node in enumerate(self.node_list):
        #     print(f'{i} node: {node.config}, cost: {node.cost}, parent: {node.parent}, children: {node.children}')
        # print('c_best: ', c_best)
        return self.get_path_to_goal()

    def distance(self, node1, node2):
        # return euclidean distance between two nodes
        return np.linalg.norm(np.array(node1.config) - np.array(node2.config))

    def sample_config(self):
        # Randomly sample joint values within specified limits
        joint_limits_min = [-2*np.pi, -2*np.pi, -np.pi]
        joint_limits_max = [2*np.pi, 2*np.pi, np.pi]
        random_config = [random.uniform(joint_min, joint_max) for joint_min, joint_max in zip(joint_limits_min, joint_limits_max)]
        
        return random_config

    def is_near_goal(self, node):
        d = self.distance(node, self.goal)
        if d < self.expand_dis:
            return True
        return False

    def find_nearest(self, rand_node, node_list):
        min_distance = float('inf')
        nearest_node = None

        for idx, node in enumerate(node_list):
            distance = self.distance(node, rand_node)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node

        return idx, nearest_node

    def steer_to(self, rand_node, nearest_node):
        # Interpolate between the two nodes with a step size of 0.05
        step_size = 0.05
        num_steps = int(1 / step_size)

        intermediate_nodes = []

        for i in range(num_steps + 1):
            t = i * step_size
            intermediate_conf = [(1-t)*nearest_node.config[j] + t*rand_node.config[j] for j in range(len(nearest_node.config))]
            if self.collision_fn(intermediate_conf):
                return False
            intermediate_nodes.append(Node(intermediate_conf))
        
        # If the entire path is collision-free, return True
        return True
        # return True, intermediate_nodes
    
    def find_near_nodes(self, newNode):
        GAMMA = 1.5

        n_node = len(self.node_list)  # number of vertices
        dimension = self.dof  # dimensions
        radius = GAMMA * (math.log(n_node)/n_node)**(1/dimension)
        # print((math.log(n_node)/n_node)**(1/dimension))
        # print('find_near_nodes radius: ', radius)
        nearinds = [i for i, node in enumerate(self.node_list) if self.distance(node, newNode) <= radius]
        
        return nearinds
    
    def update_descendant_costs(self, node):
        for child_idx in node.children:
            child_node = self.node_list[child_idx]

            # Update the cost of the descendant node
            child_node.cost = node.cost + self.distance(node, child_node)

            # Recursively update the costs of the descendant's descendants
            self.update_descendant_costs(child_node)
    
    def rewire(self, newNode, newNodeIndex, nearinds):
        for near_ind in nearinds:
            near_node = self.node_list[near_ind]

            # check for collision-free
            if self.steer_to(newNode, near_node):
                # cost from new node to near node
                cost_to_near = self.distance(newNode, near_node)
                # print(f'rewire cost {newNode.cost + cost_to_near} vs near node cost {near_node.cost}')

                # check for rewiring
                if newNode.cost + cost_to_near < near_node.cost:
                    near_node.parent = newNodeIndex
                    near_node.cost = newNode.cost + cost_to_near

                    # update the costs of the near node's descendants
                    self.update_descendant_costs(near_node)

    def choose_parent(self, newNode, nearinds):
        if not nearinds:
            return None

        min_cost = float("inf")
        parent_ind = None

        for near_ind in nearinds:
            near_node = self.node_list[near_ind]

            if self.steer_to(near_node, newNode):
                # calculate cost
                cost_to_near = near_node.cost
                cost_to_new = self.distance(near_node, newNode)
                total_cost = cost_to_near + cost_to_new
                # print(f'cost to near {cost_to_near} + cost to new {cost_to_new} = total cost {total_cost}')

                # update the parent if the total cost is smaller
                if total_cost < min_cost:
                    min_cost = total_cost
                    parent_ind = near_ind
                # print(f'min cost {min_cost}')

        return parent_ind
    
    def get_path_to_goal(self):
        """
        Traverses the tree to chart a path between the start state and the goal state.
        There may be multiple paths already discovered - if so, this returns the shortest one

        Returns: a list of coordinates, representing the path backwards; if a path has been found; None otherwise
        """
        # TODO select goal_idx with minimum dist(self.node_list[goal_idx], self.goal)
        if self.goal_found:
            goal_idx = None
            min_dist = float('inf')
            for idx in self.solution_set:
                # cost = self.node_list[idx].cost + self.distance(self.node_list[idx], self.goal)
                dist_to_goal = self.distance(self.node_list[idx], self.goal)
                if goal_idx is None or dist_to_goal < min_dist:
                    goal_idx = idx
                    min_dist = dist_to_goal
            print(f'goal idx: {goal_idx}')
            return self.gen_final_course(goal_idx)
        else:
            goal_idx = None
            min_dist = float('inf')
            for idx, node in enumerate(self.node_list):
                dist_to_goal = self.distance(node, self.goal)
                if goal_idx is None or dist_to_goal < min_dist:
                    goal_idx = idx
                    min_dist = dist_to_goal
            print(f'goal idx: {goal_idx}')
            print(f'cost: {self.node_list[goal_idx].cost}')
            return self.gen_final_course(goal_idx)
        
    def gen_final_course(self, goal_idx):
        """
        Traverses up the tree to find the path from start to goal

        goalind: index of the goal node

        Returns: a list of coordinates, representing the path backwards. Traverse this list in reverse order to follow the path from start to end
        """
        path = [self.goal.config]
        # print('path ', path)
        
        while self.node_list[goal_idx].parent is not None:
            # TODO check if parent is set correctly.. infinite loop?
            node = self.node_list[goal_idx]
            path.append(node.config)
            goal_idx = node.parent
            # print('path ', path)
        path.append(self.start.config)
        # print('** path ', path)
        return path

class Node:

    def __init__(self, config):
        self.config = config
        self.cost = 0.0
        self.parent = None   # index of parent node in the node_list
        self.children = []   # indices of children nodes in the node_list

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        p.resetJointState(body, joint, value)


if __name__ == '__main__':
    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    p.resetDebugVisualizerCamera(cameraDistance=1.400, cameraYaw=58.000, cameraPitch=-42.200, cameraTargetPosition=(0.0, 0.0, 0.0))

    # load objects
    plane = p.loadURDF("plane.urdf")
    ur5 = p.loadURDF('informed_rrtstar/assets/ur5/ur5.urdf', basePosition=[0, 0, 0.02], useFixedBase=True)
    obstacle1 = p.loadURDF('informed_rrtstar/assets/block.urdf',
                           basePosition=[1/4, 0, 1/2],
                           useFixedBase=True)
    obstacle2 = p.loadURDF('informed_rrtstar/assets/block.urdf',
                           basePosition=[2/4, 0, 2/3],
                           useFixedBase=True)
    obstacles = [plane, obstacle1, obstacle2]

    UR5_JOINT_INDICES = [0, 1, 2]
    start_conf = np.array([-0.813358794499552, -0.37120422397572495, -0.754454729356351])
    start_position = (0.3998897969722748, -0.3993956744670868, 0.6173484325408936)
    goal_conf = np.array([0.7527214782907734, -0.6521867735052328, -0.4949270744967443])
    goal_position = (0.35317009687423706, 0.35294029116630554, 0.7246701717376709)
    set_joint_positions(ur5, UR5_JOINT_INDICES, start_conf)

    print('q_init: ', start_conf)
    print('q_goal: ', goal_conf)

    # get the collision checking function
    collision_fn = get_collision_fn(ur5, UR5_JOINT_INDICES, obstacles=obstacles,
                                       attachments=[], self_collisions=True,
                                       disabled_collisions=set())

    time.sleep(1)

    path = None
    print("Start informed rrt star planning")
    planner = InformedRRTStar(start_config=start_conf, goal_config=goal_conf, obstacle_list=obstacles, 
                              ur5=ur5, UR5_JOINT_INDICES=UR5_JOINT_INDICES)
    path = planner.plan()
    
    if path is None:
        # pause here
        input("no collision-free path is found within the time budget, finish?")
    else:
        print('Found the path!')
        print(path)
        # execute the path
        while True:
            for q in path:
                set_joint_positions(ur5, UR5_JOINT_INDICES, q)
                time.sleep(1)