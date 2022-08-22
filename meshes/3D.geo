// ---------------------------------------------------------------------
// Copyright (C) 2022 by the DUBeat authors.
//
// This file is part of DUBeat.
//
// DUBeat is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DUBeat is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DUBeat.  If not, see <http://www.gnu.org/licenses/>.
// ---------------------------------------------------------------------

// Author: Matteo Calafà <matteo.calafa@mail.polimi.it>.

// This .geo file can be used to generate on gmsh a simple cube mesh with a specific refinement level.
// The meshes contained in the folder are created using N as powers of 2.
// For instance, 3D_4 is generated from this file using N=2^(4-1)=8.

// Parameter to modify ( = Cube edge / element edge ratio)
N = 1;

SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Plane Surface(1) = {1};
//+
Extrude {0, 0, 1} {
  Surface{1};
}
//+
Transfinite Curve {7, 9, 11, 12, 10, 5, 2, 1, 3, 4, 6, 8} = N+1 Using Progression 1;
//+
Transfinite Surface {6};
//+
Transfinite Surface {5};
//+
Transfinite Surface {1};
//+
Transfinite Surface {3};
//+
Transfinite Surface {4};
//+
Transfinite Surface {2};
//+
Transfinite Volume{1} = {7, 8, 5, 6, 1, 2, 3, 4};
