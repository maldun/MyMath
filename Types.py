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
import numpy as np

class MathOperator(object):
    """
    Base class for operators. An operator is
    characterized by the fact that it has a call
    method on vectors.
    """

    def _pythonOP(self,x):
        """
        Method for math operation in pure Python.
        """
        raise NotImplementedError("Error: This class has no Python method!")
    
    def _optimizedOP(self,x):
        """
        Optimized (optional) method for math operation.
        """
        raise NotImplementedError("Error: This class has no optimized method!")
        
    def __call__(self,x):
        """
        General call method for math operators.
        """
        try:
            return self._optimizedOP(x)
        except NotImplementedError:
            return self._pythonOP(x)

class GeometricTransformation(MathOperator):
    u"""
    Represents a geometric transformation T of the form

    T(x) = Qx + b

    where Q ∈ OS₃(ℝ) and b ∈ ℝ³. 
    """

    def __init__(self,Q,b):

        self.Q = Q
        dimension = b.size
        self.b = b.reshape((dimension,1))

    def _pythonOP(self,x):

        return np.dot(self.Q,x) + self.b

    
