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
        from Types import PolarCoordinates
        pi = np.pi
        from numpy import sqrt, cos, sin
        
        passed = [False]
        with warnings.catch_warnings(record=True) as warn:
            polarDummy = PolarCoordinates(method=666)
            assert issubclass(warn[-1].category, UserWarning)
            assert self.fallback_warning in str(warn[-1].message)
        passed[0] = True
        polar = PolarCoordinates(method=0)
        # test special cases
        v = np.array([0.0,1.0]); w = np.array([1.0,pi/2]) 
        assert norm(polar(v) - w) < eps, "Error: Special case 1 false!"
        v = np.array([0.0,-2.0]); w = np.array([2.0,-pi/2])
        assert norm(polar(v) - w) < eps, "Error: Special case 2 false!"
        passed += [True]
        # first sector 
        v = np.array([sqrt(0.5),sqrt(0.5)]); w = np.array([1.0,pi/4])
        assert norm(polar(v) - w) < eps, "Error: Sector 1 false!"
        v = np.array([3*cos(7*pi/8),3*sin(7*pi/8)]); w = np.array([3.0,7*pi/8])
        assert norm(polar(v) - w) < eps, "Error: Sector 2 false!"
        v = np.array([5*cos(-7*pi/8),5*sin(-7*pi/8)]); w = np.array([5.0,-7*pi/8])
        assert norm(polar(v) - w) < eps, "Error: Sector 3 false!"
        v = np.array([8*cos(-pi/5),8*sin(-pi/5)]); w = np.array([8.0,-pi/5])
        assert norm(polar(v) - w) < eps, "Error: Sector 4 false!"

        nr_tests = self.nr_tests
        R = np.random.rand(nr_tests)*nr_tests
        PHI = 2*pi*(np.random.rand(nr_tests)-0.5)
        for i in range(nr_tests):
            v = np.array([R[i]*cos(PHI[i]),R[i]*sin(PHI[i])]); w = np.array([R[i],PHI[i]])
            assert norm(polar(v) - w) < eps, "Error: Random test" + str(i) + " false!"
        
        self.checkTests("PolarCoordinates",passed)

    def testCartesianCoordinates(self):
        from Types import CartesianCoordinates
        pi = np.pi
        from numpy import sqrt, cos, sin
        
        passed = [False]
        with warnings.catch_warnings(record=True) as warn:
            cartesianDummy = CartesianCoordinates(method=666)
            assert issubclass(warn[-1].category, UserWarning)
            assert self.fallback_warning in str(warn[-1].message)
        passed[0] = True
        cart = CartesianCoordinates(method=0)
        # test wron Input
        try:
            cart(np.array((5,7)))
        except ValueError:
            passed += [True]
        
        # test special cases
        v = np.array([0.0,1.0]); w = np.array([1.0,pi/2]) 
        assert norm(cart(w) - v) < eps, "Error: Special case 1 false!"
        v = np.array([0.0,-2.0]); w = np.array([2.0,-pi/2])
        assert norm(cart(w) - v) < eps, "Error: Special case 2 false!"
        passed += [True]
        # first sector 
        v = np.array([sqrt(0.5),sqrt(0.5)]); w = np.array([1.0,pi/4])
        assert norm(cart(w) - v) < eps, "Error: Sector 1 false!"
        v = np.array([3*cos(7*pi/8),3*sin(7*pi/8)]); w = np.array([3.0,7*pi/8])
        assert norm(cart(w) - v) < eps, "Error: Sector 2 false!"
        v = np.array([5*cos(-7*pi/8),5*sin(-7*pi/8)]); w = np.array([5.0,-7*pi/8])
        assert norm(cart(w) - v) < eps, "Error: Sector 3 false!"
        v = np.array([8*cos(-pi/5),8*sin(-pi/5)]); w = np.array([8.0,-pi/5])
        assert norm(cart(w) - v) < eps, "Error: Sector 4 false!"

        nr_tests = self.nr_tests
        R = np.random.rand(nr_tests)*nr_tests
        PHI = 2*pi*(np.random.rand(nr_tests)-0.5)
        for i in range(nr_tests):
            v = np.array([R[i]*cos(PHI[i]),R[i]*sin(PHI[i])]); w = np.array([R[i],PHI[i]])
            assert norm(cart(w) - v) < eps, "Error: Random test" + str(i) + " false!"
        
        self.checkTests("CartesianCoordinates",passed)

    def testSphericalCoordinates(self):
        from Types import SphericalCoordinates
        pi = np.pi
        from numpy import sqrt, cos, sin
        passed = [False]
        with warnings.catch_warnings(record=True) as warn:
            sphericalDummy = SphericalCoordinates(method=666)
            assert issubclass(warn[-1].category, UserWarning)
            assert self.fallback_warning in str(warn[-1].message)
        passed[0] = True
        
        spher = SphericalCoordinates(method=0)
        nr_tests = self.nr_tests
        R = np.random.rand(nr_tests)*nr_tests
        PHI = 2*pi*(np.random.rand(nr_tests)-0.5)
        PSI = pi*(np.random.rand(nr_tests))
        for i in range(nr_tests):
            v = np.array([R[i]*cos(PHI[i])*sin(PSI[i]),
                          R[i]*sin(PHI[i])*sin(PSI[i]),
                          R[i]*cos(PSI[i])]) 
            w = np.array([R[i],PHI[i],PSI[i]])
            assert norm(spher(v) - w)  < eps, "Error: Random test" + str(i) + " false!"
        passed += [True]
            
        self.checkTests("SphericalCoordinates",passed)


    def testCartesianCoordinates3D(self):
        from Types import CartesianCoordinates3D
        pi = np.pi
        from numpy import sqrt, cos, sin
        
        passed = [False]
        with warnings.catch_warnings(record=True) as warn:
            cartesianDummy = CartesianCoordinates3D(method=666)
            assert issubclass(warn[-1].category, UserWarning)
            assert self.fallback_warning in str(warn[-1].message)
        passed[0] = True
        cart = CartesianCoordinates3D(method=0)
        # test wron Input
        try:
            cart(np.array((5,7,0.1)))
        except ValueError:
            passed += [True]
        try:
            cart(np.array((5,0.1,5)))
        except ValueError:
            passed += [True]
        
        nr_tests = self.nr_tests
        R = np.random.rand(nr_tests)*nr_tests
        PHI = 2*pi*(np.random.rand(nr_tests)-0.5)
        PSI = pi*(np.random.rand(nr_tests))
        for i in range(nr_tests):
            v = np.array([R[i]*cos(PHI[i])*sin(PSI[i]),
                          R[i]*sin(PHI[i])*sin(PSI[i]),
                          R[i]*cos(PSI[i])]) 
            w = np.array([R[i],PHI[i],PSI[i]])
            assert norm(cart(w) - v)  < eps, "Error: Random test" + str(i) + " false!"

        self.checkTests("CartesianCoordinates3D",passed)


    def testGivensRotator(self):

        from Types import GivensRotator
        pi = np.pi
        from numpy import sqrt, cos, sin, abs
        passed = [False]
        # test creation
        with warnings.catch_warnings(record=True) as warn:
            rotDummy = GivensRotator(0,1,0.0,method=666)
            assert issubclass(warn[-1].category, UserWarning)
            assert self.fallback_warning in str(warn[-1].message)
        passed[0] = True

        # Check creation
        rot1 = GivensRotator(0,1,0.0)
        assert abs(rot1.c-1.0) < eps
        assert abs(rot1.s) < eps
        rot2 = GivensRotator(0,1,pi/2)
        assert abs(rot2.c-0.0) < eps
        assert abs(rot2.s-1.0) < eps
        rot3 = GivensRotator(0,1,sqrt(0.5),sqrt(0.5))
        assert abs(rot3.c-sqrt(0.5)) < eps
        assert abs(rot3.s-sqrt(0.5)) < eps
        passed += [True]
        try:
            GivensRotator(0,1,0.5,0.5)
            passed += [False]
        except ValueError:
            passed += [True]
        # test if phi is computd correctly 
        assert abs(rot3.getPhi()-pi/4) < eps
        self.checkTests("GivensRotator",passed)

    def __init__(self):
        """
        Method for executing tests.
        """
        self.fallback_warning = "Warning: Make fallback since no other version is implemented!"
        self.nr_tests = 50
        
        self.testPolarCoordinates()
        self.testCartesianCoordinates()
        self.testSphericalCoordinates()
        self.testCartesianCoordinates3D()
        self.testGivensRotator()
        
TypeTests2()
