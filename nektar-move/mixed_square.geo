// Bigger square points
Point(1) = {-1.00, -1.00,   0.00, 0.1};
Point(2) = { 1.00, -1.00,   0.00, 0.1};
Point(3) = { 1.00,  1.00,   0.00, 0.1};
Point(4) = {-1.00,  1.00,   0.00, 0.1};

// Refinement region points
Point(5) = { 1.00, -0.25,   0.00, 0.1};
Point(6) = { 1.00,  0.25,   0.00, 0.1};
Point(7) = {-1.00,  0.25,   0.00, 0.1};
Point(8) = {-1.00, -0.25,   0.00, 0.1};

// Lines of the square
Line(1)  = {1, 2};
Line(2)  = {2, 5};
Line(3)  = {5, 6};
Line(4)  = {6, 3};
Line(5)  = {3, 4};
Line(6)  = {4, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 1};

// Two central lines for mesh generation purposes
Line(9)  = {5, 8};
Line(10) = {6, 7};

//Line Loops to create physical surfaces
Line Loop(1)        = {1, 2, 9, 8};
Plane Surface(1)    = {1};
Physical Surface(1) = {1};

Line Loop(2)        = {-9, 3, 10, 7};
Plane Surface(2)    = {2};
Physical Surface(2) = {2};

Line Loop(3)        = {-10, 4, 5, 6};
Plane Surface(3)    = {3};
Physical Surface(3) = {3};

// Making a structured mesh in the central region
Transfinite Line {7, -3} = 5 Using Progression 1;
Transfinite Surface{2};
Recombine Surface{2};

// Making a structured mesh in the other two regions
//Transfinite Line {2, -8} = 5 Using Progression 0.75;
//Transfinite Surface{1};
//Recombine Surface{1};
//Transfinite Line {4, -6} = 5 Using Progression 1.25;
//Transfinite Surface{3};
//Recombine Surface{3};

// Defining physical lines for applying the boundary
// conditions
Physical Line(100) = {1};
Physical Line(200) = {2,3,4};
Physical Line(300) = {5};
Physical Line(400) = {6,7,8};
