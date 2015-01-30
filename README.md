# MyMath
Math tool collection for use with MyGeom and MyMesh

Author: Stefan Reiterer Email: stefan.harald.reiterer@gmail.com

License:

Copyright (C) 2015 Stefan Reiterer - stefan.harald.reiterer@gmail.com

This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version. This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You should have received a copy of the GNU Lesser General Public License along with this library; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

(same License as Salome)

## Discription
This Module provides mathematical tools for vector operations for usage with MyGeom and MyMesh modules.

## Design
The goal is to provide functions and classes for common mathematical operations. 
All operations come either in pure Python or with an optional optimized version.
If no optimized version can is available the pure Python version is used instead, else
the optimized version is used.
The Python version has to provided everytime for compability and testing reasons,
or even for performance reasons.
Each instance tries to find the most effective wy for computation and implements it
on runtime. 
