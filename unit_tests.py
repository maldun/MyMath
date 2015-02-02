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

"""
Unit tests for the MyMath module
"""
from __future__ import print_function
import warnings
import numpy as np
from numpy import ones, eye, dot
from numpy.linalg import norm
eps = 10*np.finfo(np.float32).eps
warnings.simplefilter('always', UserWarning)

class TypeTests(object):
    """
    This class contains the unit tests for the Type module
    """
    
    def checkTests(self,name,passed):
        """
        Checks if all tests of a test with name failed
        """
        if all(passed):
            print(name + " tests passed!")
        else:
            raise Exception(name + " tests failed!")


    def testMathOperator(self):
        """
        Base Class
        """
        from Types import MathOperator
        passed = [False]
        x = 5
        # init
        
        try:
            with warnings.catch_warnings(record=True) as warn:
                mathi = MathOperator()
                assert self.fallback_warning in str(warn[-1].message)
                try:
                    mathi(x)
                except NotImplementedError:
                    passed[0] = True
        except AssertionError:
            pass


        try:
            mathi._pythonOP(x)
        except NotImplementedError:
            passed += [True]

        try:
            mathi._optimizedOP(x)
        except NotImplementedError:
            passed += [True]

        self.checkTests('MathOperator',passed)

    def testGeometricTransformation(self):
        """
        testing the Geometric Transformation class
        """
        from Types import GeometricTransformation
        passed = [False]
        # Tests for Init
        # test if b is a vector
        try:
            GeometricTransformation(ones((3,3)),ones((3,3)))
        except ValueError:
            passed[0] = True

        # test if Q is orthogonal works
        try:
            GeometricTransformation(ones((3,3)),ones(3))
            passed += [False]
        except ValueError:
            passed += [True]

        # test if fallback works
        with warnings.catch_warnings(record=True) as warn:
            mathi = GeometricTransformation(eye(3),ones(3),method=666)
            assert self.fallback_warning in str(warn[-1].message)
            passed += [True]
        
        # Mathematical correctness:
        # special case
        tester = GeometricTransformation(eye(3),ones(3))
        assert norm(tester(ones(3))-ones(3)*2) < eps
        
        # series of random tests
        for i in range(10):
            QR = np.random.randn(3,3)
            QR = np.linalg.qr(QR)[0]
            bR = np.random.randn(3,1)
            testerR = GeometricTransformation(QR,bR)
            xR = np.random.randn(3,1)
            assert norm(testerR(xR) - dot(QR,xR) - bR) < eps
            testerR_inv = testerR.inv()
            assert norm(testerR_inv(testerR(xR)) - xR) < eps
        
        passed += [True]
        self.checkTests('GeometricTransformation',passed)

    def __init__(self):
        """
        Init method runs all the tests
        """
        self.fallback_warning = "Warning: Make fallback since no other version is implemented!"
        self.testMathOperator()
        self.testGeometricTransformation()

# run tests
TypeTests()
