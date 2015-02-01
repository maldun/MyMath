#!/usr/bin/python
# -*- coding: utf-8 -*-

# MyMath Module - API for easier Salome geompy usage
# Tools.py: Math functions and operations for MyMath module
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
eps = 10*np.finfo(np.float32).eps
import Types

def givens_qr(A,transposed = False):
    """
    Computes the QR decomposition of
    a matrix A such that Q*R = A
    If transposed is True Q.transpose()
    is returned
    """
    if len(A.shape) != 2:
        raise np.LinAlgError
    dim = A.shape[0]
    if dim is 0 or dim is 1:
        warnings.warn("Warning: Dimension < 2! Q is scalar not GivensRotation!",UserWarning)
        return np.array([1.0]), A
        
    Q = Types.GivensRotations(dim=dim)
    result = A
    for i in range(dim-1):
        for j in range(i+1,dim):
            result = Q.computeRotation(i,j,result)
            
    R = result
    if transposed:
        return Q, R
    else:
        return Q.transpose(), R 
