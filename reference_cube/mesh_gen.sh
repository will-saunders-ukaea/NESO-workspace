BASENAME=$(basename $1 .geo)
gmsh -3 $1
NekMesh-rg -v -m peralign:surf1=100:surf2=200:dir=x:orient -m peralign:surf1=300:surf2=400:dir=y:orient -m peralign:surf1=500:surf2=600:dir=z:orient "$BASENAME.msh" "$BASENAME.tmp.xml":xml:uncompress

awk '!/EXPANSIONS/' "$BASENAME.tmp.xml" > "$BASENAME.tmp2.xml"
rm "$BASENAME.tmp.xml"
awk '!/NUMMODES/' "$BASENAME.tmp2.xml" > "$BASENAME.xml"
rm "$BASENAME.tmp2.xml"
