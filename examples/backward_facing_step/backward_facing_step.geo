// Adjustable coarse/fine proxy for the backward-facing-step tutorial.
// Generate with: gmsh -2 backward_facing_step_proxy.geo -format msh2

SetFactory("Built-in");

H = 1.0;
L_up = 2.0;
L_down = 6.0;

// Primary mesh controls. Reduce these values for a finer mesh.
h_wall = 0.2;
h_core = 0.2;

Point(1) = {-L_up,   H,     0, h_wall};
Point(2) = {0,       H,     0, h_wall};
Point(3) = {0,       0,     0, h_wall};
Point(4) = {L_down,  0,     0, h_wall};
Point(5) = {L_down,  2 * H, 0, h_wall};
Point(6) = {-L_up,   2 * H, 0, h_wall};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};

Physical Curve("inlet") = {6};
Physical Curve("outlet") = {4};
Physical Curve("wall") = {1, 2, 3, 5};
Physical Surface("surface") = {1};

Field[1] = Distance;
Field[1].CurvesList = {1, 2, 3, 5};
Field[1].Sampling = 250;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_wall;
Field[2].SizeMax = h_core;
Field[2].DistMin = 0.10 * H;
Field[2].DistMax = 0.50 * H;
Background Field = 2;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 2.2;
