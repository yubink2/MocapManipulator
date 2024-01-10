from __future__ import print_function
import argparse
import math
import numpy as np
from ConstrainedPlanningCommon import *
from tsr import *

import sympy as sp


class TSRConstraint(ob.Constraint):
    def __init__(self, tsr_chain):
        self.tsr_chain = tsr_chain
        super(TSRConstraint, self).__init__(3, 1)

    def function(self, x, out):
        ?

    def jacobian(self, x, out):
        return self.function(x, out).jacobian(x)
        # return self.getConstraints().jacobian(self.variables_)
