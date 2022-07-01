// Gmsh project created on Mon Feb 14 17:39:09 2022

// PARAMETER TO MODIFY (CUBE EDGE / ELEMENT EDGE RATIO)
N = 0;

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
