BASENAME=$(basename $1 .geo)
gmsh -2 $1
NekMesh-rg -v -m peralign:surf1=100:surf2=300:dir=y -m peralign:surf1=200:surf2=400:dir=x "$BASENAME.msh" "$BASENAME.tmp.xml":xml:uncompress

awk '!/EXPANSIONS/' "$BASENAME.tmp.xml" > "$BASENAME.tmp2.xml"
rm "$BASENAME.tmp.xml"
awk '!/NUMMODES/' "$BASENAME.tmp2.xml" > "$BASENAME.xml"
rm "$BASENAME.tmp2.xml"
