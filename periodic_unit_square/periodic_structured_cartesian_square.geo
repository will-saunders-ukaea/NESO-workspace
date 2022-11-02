// Bigger square points
Point(1) = {0.00, 0.00,   0.00, 0.20};
Point(2) = {1.00, 0.00,   0.00, 0.20};
Point(3) = {1.00, 1.00,   0.00, 0.20};
Point(4) = {0.00, 1.00,   0.00, 0.20};


// Lines of the square
Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};


//Line Loops to create physical surfaces
Line Loop(1)        = {1, 2, 3, 4};
Plane Surface(1)    = {1};
Physical Surface(1) = {1};

// Making a structured mesh in the central region
Transfinite Line {7, -3} = 5 Using Progression 1;
Transfinite Surface{1};
Recombine Surface{1};

Periodic Curve {3} = {1};
Periodic Curve {4} = {2};

// Defining physical lines for applying the boundary
// conditions
Physical Line(100) = {1};
Physical Line(200) = {2};
Physical Line(300) = {3};
Physical Line(400) = {4};
