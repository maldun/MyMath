#!/usr/bin/python
# -*- coding: utf-8 -*-

# MyGeom Module - API for easier Salome geompy usage
# Tools.py: Tool functions for MyGeom module
#
# Copyright (C) 2015  Stefan Reiterer - stefan.harald.reiterer@gmail.com
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

from __future__ import print_function
import warnings
import numpy as np
from numpy import ones, dot, eye
from numpy.linalg import norm
eps = 10*np.finfo(np.float32).eps
import warnings
warnings.simplefilter('always', UserWarning)

# This second type test class is created due to logistical reasons ....
class TypeTests2(object):
    """
    Second test class for the Type module.
    """

    def checkTests(self,name,passed):
        if not all(passed):
            raise Exception("Error: " + name + " tests didn't passed!")
        print("All " + name + " tests passed!")
    
    def testPolarCoordinates(self):

        passed = [False]
        
    
    def __init__(self):
        """
        Method for executing tests.
        """
        self.testPolarCoordinates()


TypeTests2()
