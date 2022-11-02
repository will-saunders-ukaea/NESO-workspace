gmsh -2 periodic_structured_cartesian_square.geo
NekMesh-rg -v -m peralign:surf1=100:surf2=300:dir=y -m peralign:surf1=200:surf2=400:dir=x periodic_structured_cartesian_square.msh Mesh_aligned.xml:xml:uncompress
