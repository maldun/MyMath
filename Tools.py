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

def _py_givens_qr(A,transposed = False,copy=True):

# function with method selection
def givens_qr(A,transposed = False,copy=True,method=0):
    
    return _py_givens_qr(A,transposed,copy,method)
