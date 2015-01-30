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
eps = 10*np.finfo(np.float32).eps

class MathOperator(object):
    u"""
    Base class for operators. An operator is
    characterized by the fact that it has a call
    method on vectors.
    """

    def _pythonOP(self,x):
        """
        Method for math operation in pure Python.
        """
        raise NotImplementedError("Error: This class has no Python method!")
    
    def __call__(self,x):
        return self._operation(x)

    def _fallback(self):
        """
        If not other functions are implemented make fallback to Python.
        """
        warnings.warn("Warning: Make fallback since no other version is implemented!",UserWarning)
        self._operation = self._pythonOP

    def _operation(self,x):
        raise NotImplementedError("Error: The operation is not set yet!")

    def _optimizedOP(self,x):
        """
        Optimized (optional) method for math operation.
        """
        raise NotImplementedError("Error: This class has no optimized method!")
        
    def __init__(self,method=0):
        """
        This __init__ method is reference only to
        provide a proper example for further classes.
        params: 
            method (Integer):
            method = 0: use Python
            method = 1: Use optimized version
            method > 1: Use other alternative methods
        """
        self._fallback()

class GeometricTransformation(MathOperator):
    u"""
    Represents a geometric transformation T of the form

    T(x) = Qx + b

    where Q ∈ OS₃(ℝ) and b ∈ ℝ³. 
    Vectors are assumed to be column vectors.
    If they are not they get reshaped.
    """

    def __init__(self,Q,b,method=0):

        if (len(b.shape) != 1) and (b.shape[-1] != 1):
            raise ValueError("Error: Input vector b is not a vector!")
        dimension = b.size
        if not np.linalg.norm(np.dot(Q.transpose(),Q) - np.eye(dimension)) < eps:
            raise ValueError("Error: Q not Orthogonal!")
        
        self.Q = Q
        self.b = b.reshape((dimension,1))

        # If exception in creation caught make fallback
        try:
            if method is 0:
                self._operation = self._pythonOP
            else:
                raise NotImplementedError("Error: No other methods implemented yet!")
        except:
            self._fallback()
            
    def _pythonOP(self,x):

        if len(x.shape) == 1:
            result = x.reshape((x.size,1))
        else: 
            result = x
        return np.dot(self.Q,result) + self.b

    def inv(self):
        """
        Returns the inverse transformation S of T
        such that S(T(x)) = x for all x.
        """
        Q_inv = (self.Q).transpose()
        b_inv = -np.dot(Q_inv,self.b)
        return GeometricTransformation(Q_inv,b_inv)

class PolarCoordinates(MathOperator):
    u"""
    Takes a vector v ∈ ℝ² given in
    cartesian coordintes v₁ = x,
    v₂ = y, and returns
    a vector w ∈ ℝ² given in
    polar coorinates 
    w₁ = r, w₂ = φ with r > 0,
    and φ ∈ [-π,π].
    """

    def __init__(self,method=0):

        try:
            if method is 0:
                self._operation = self._pythonOP
            else:
                raise NotImplementedError("Error: No other methods implemented yet!")
        except:
            self._fallback()
 
        
    def _pythonOP(self,v):
        """
        Computes the polar coordinates,
        by the well known formulas.
        We arange the plane in 4 sectors:
         y
         ^
        2|1
        ---> x
        3|4
        """
        if v.size != 2:
            raise ValueError("Error: Input vector has wrong dimension!")
        
        x = v[0]
        y = v[1]
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y,x)

        return np.array([r,phi])

class CartesianCoordinates(MathOperator):
    u"""
    Takes a vector w ∈ ℝ² given in
    polar coordintes w₁ = r, 
    w₂ = φ with r > 0, and φ ∈ [-π,π],
    and a vector w ∈ ℝ² given in
    cartesian coorinates
    v₁ = x, v₂ = y is returned.
    """
    def __init__(self,method=0):

        try:
            if method is 0:
                self._operation = self._pythonOP
            else:
                raise NotImplementedError("Error: No other methods implemented yet!")
        except:
            self._fallback()

    def _pythonOP(self,w):
        """
        This method computes the
        cartesian coordinates by the formulas 
        x = r cos(φ),
        y = r sin(φ).
        """
        if w.size != 2:
            raise ValueError("Error: Input vector has wrong dimension!")

        r = w[0]
        phi = w[1]
        # check data range:
        if not (-np.pi <= phi <= np.pi):
            raise ValueError(u"""Error: φ ∉ [-π,π]""")
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        return np.array([x,y])
    
class SphericalCoordinates(MathOperator):
    u"""
    Takes a vector v ∈ ℝ³ given in
    cartesian coordintes v₁ = x,
    v₂ = y, v₃ = z and returns
    a vector w ∈ ℝ³ given in
    spherical coorinates 
    w₁ = r, w₂ = φ, w₃ = ψ,
    with φ ∈ [-π,π] and
    ψ ∈ [0,π].
    """
    def __init__(self,method=0):

        try:
            if method is 0:
                self._operation = self._pythonOP
            else:
                raise NotImplementedError("Error: No other methods implemented yet!")
        except:
            self._fallback()
    
    def _pythonOP(self,v):
        if v.size != 3:
            raise ValueError("Error: Dimension is not 3!")

        x = v[0]
        y = v[1]
        z = v[2]
        r = np.linalg.norm(v)
        phi = np.arctan2(y,x)
        psi = np.arccos(z/r)

        return np.array([r,phi,psi])

class CartesianCoordinates3D(MathOperator):
    u"""
    Takes a vector w ∈ ℝ³ given in
    spherical coorinates 
    w₁ = r, w₂ = φ, w₃ = ψ,
    with φ ∈ [-π,π] and
    ψ ∈ [0,π], and returns a 
    vector v ∈ ℝ³ given in
    cartesian coordintes v₁ = x,
    v₂ = y, v₃ = z. 
    """
    
    def __init__(self,method=0):

        try:
            if method is 0:
                self._operation = self._pythonOP
            else:
                raise NotImplementedError("Error: No other methods implemented yet!")
        except:
            self._fallback()
    
    def _pythonOP(self,w):
        u"""
        This method computes the
        cartesian coordinates by the formulas 
        x = r cos(φ)sin(ψ),
        y = r sin(φ)sin(ψ),
        z = r cos(ψ).
        """

        if w.size != 3:
            raise ValueError("Error: Dimension is not 3!")
        r = w[0]
        phi = w[1]
        psi = w[2]
        # check data range:
        if not (-np.pi <= phi <= np.pi):
            raise ValueError(u"""Error: φ ∉ [-π,π]""")
        if not (0 <= psi <= np.pi):
            raise ValueError(u"""Error: ψ ∉ [0,π]""")

        
        x = r*np.cos(phi)*np.sin(psi)
        y = r*np.sin(phi)*np.sin(psi)
        z = r*np.cos(psi)

        return np.array([x,y,z])
