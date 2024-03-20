from __future__ import print_function
import argparse
import math
import numpy as np
from ConstrainedPlanningCommon import *
from tsr import *

import sympy as sp

# robot.arm_controllable_joints = 12
# virtualManipDof = 5

class TSRConstraint(ob.Constraint):
    def __init__(self, tsr_chain, robot, virtualManipDOF, virtualHumanoid, bc_second):
        self.tsr_chain = tsr_chain
        self.robot = robot
        self.virtualManipDOF = virtualManipDOF
        self.virtualHumanoid, virtualHumanoid
        self.bc_second = bc_second

        # super(TSRConstraint, self).__init__(self.robot.arm_controllable_joints + virtualManipDOF, 1)
        super(TSRConstraint, self).__init__(12+5 1)

    def function(self, x, out):
        ?

    def jacobian(self, x, out):
        return self.function(x, out).jacobian(x)
        # return self.getConstraints().jacobian(self.variables_)

class TSRProjection(ob.ProjectionEvaluator):

    def __init__(self, space):
        super(TSRProjection, self).__init__(space)

    def getDimension(self):
        return 2

    def defaultCellSizes(self):
        self.cellSizes_ = list2vec([.1, .1])

    def project(self, state, projection):
        projection[0] = math.atan2(state[1], state[0])
        projection[1] = math.acos(state[2])

def TSRPlanning(options, robot_dof, tsr_dof, robot):
    # Create the ambient space state space for the problem.
    rvss = ob.RealVectorStateSpace(robot_dof + tsr_dof)

    # Set bounds on the space.
    bounds = ob.RealVectorBounds(robot_dof + tsr_dof)

    # get bounds for real robot
    bounds[:robot_dof, 0] = robot.arm_lower_limits[:robot_dof]
    bounds[:robot_dof, 1] = robot.arm_upper_limits[:robot_dof]

    # get bounds for virtual robot
    offset = robot_dof
    for tsrchain in tsrchains_:
        dof_tsr = tsrchain.GetNumDOF()
        lower_limits, upper_limits = tsrchain.GetChainJointLimits()
        bounds[offset:offset+dof_tsr, 0] = lower_limits[:dof_tsr]
        bounds[offset:offset+dof_tsr, 1] = upper_limits[:dof_tsr]
        offset += dof_tsr

    rvss.setBounds(bounds)

    # Create our constraint.
    constraint = TSRConstraint()

    cp = ConstrainedProblem(options.space, rvss, constraint, options)
    cp.css.registerProjection("TSR", TSRProjection(cp.css))

    start = ob.State(cp.css)
    goal = ob.State(cp.css)
    start[0] = 0
    start[1] = 0
    start[2] = -1
    goal[0] = 0
    goal[1] = 0
    goal[2] = 1
    cp.setStartAndGoalStates(start, goal)
    cp.ss.setStateValidityChecker(ob.StateValidityCheckerFn(obstacles))

    cp.setPlanner(plannername, "TSR")

    # Solve the problem
    stat = cp.solveOnce(output, "TSR")  

    return stat


# constraint function:
# move virtual manip close to real robot

# move real robot close to virtual manip