
BASENAME=$(basename $1 .geo)
gmsh -2 $1
NekMesh-rg -v -m peralign:surf1=100:surf2=300:dir=y -m peralign:surf1=200:surf2=400:dir=x "$BASENAME.msh" "$BASENAME.xml":xml:uncompress
