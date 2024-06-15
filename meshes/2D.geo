// ---------------------------------------------------------------------
// Copyright (C) 2024 by the DUBeat authors.
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
// For instance, 2D_4 is generated from this file using N=2^(4-1)=8.

SetFactory("OpenCASCADE");

// Parameter to modify ( = Cube edge / element edge ratio)
N = 1;

Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Transfinite Curve {4, 1, 2, 3} = N+1 Using Progression 1;
//+
Transfinite Surface {1};
