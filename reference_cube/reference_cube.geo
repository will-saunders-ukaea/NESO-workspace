SetFactory("OpenCASCADE");

Box(1) = {-1, -1, -1, 2, 2, 2};
MeshSize {:} = 0.5;
Periodic Surface {2} = {1} Translate {2, 0, 0};
Periodic Surface {6} = {5} Translate {0, 0, 2};
Periodic Surface {4} = {3} Translate {0, 2, 0};

Physical Surface (100) = {1};
Physical Surface (200) = {2};
Physical Surface (300) = {3};
Physical Surface (400) = {4};
Physical Surface (500) = {5};
Physical Surface (600) = {6};

Physical Volume (1) = {1};
