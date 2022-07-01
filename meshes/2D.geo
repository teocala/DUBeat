// Gmsh project created on Thu Jan  6 18:54:25 2022
SetFactory("OpenCASCADE");

// PARAMETER TO MODIFY (CUBE EDGE / ELEMENT EDGE RATIO)
N = 64;

Rectangle(1) = {0, 0, 0, 1, 1, 0};
//+
Transfinite Curve {4, 1, 2, 3} = N+1 Using Progression 1;
//+
Transfinite Surface {1};
