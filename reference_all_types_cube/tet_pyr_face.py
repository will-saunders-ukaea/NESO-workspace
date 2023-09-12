import gmsh
import sys
import numpy as np

# small values generate a more refined mesh
lc = 0.5
# set to False to disable pyramid generation at bottom face
make_pyr = True

class EntityMap:
    def __init__(self, next_value=1):
        self.map = {}
        self.next = next_value

    def __getitem__(self, key):
        if key not in self.map.keys():
            self.map[key] = self.next
            self.next += 1
        return self.map[key]


pm = EntityMap()
lm = EntityMap()
cm = EntityMap()
sm = EntityMap()
slm = EntityMap()
vm = EntityMap()

# If sys.argv is passed to gmsh.initialize(), Gmsh will parse the command line
# in the same way as the standalone Gmsh app:
gmsh.initialize(sys.argv)
gmsh.model.add("ref_cube")

zlevels = [-1.0, 1.0]
nlevels = len(zlevels)

# add the horizontal planes
for z, zc in enumerate(zlevels):
    gmsh.model.geo.addPoint(-1.0, -1.0, zc, lc, pm[f"{z}SW"])
    gmsh.model.geo.addPoint(1.0, -1.0, zc, lc, pm[f"{z}SE"])
    gmsh.model.geo.addPoint(1.0, 1.0, zc, lc, pm[f"{z}NE"])
    gmsh.model.geo.addPoint(-1.0, 1.0, zc, lc, pm[f"{z}NW"])
    gmsh.model.geo.addLine(pm[f"{z}SW"], pm[f"{z}SE"], lm[f"{z}S"])
    gmsh.model.geo.addLine(pm[f"{z}SE"], pm[f"{z}NE"], lm[f"{z}E"])
    gmsh.model.geo.addLine(pm[f"{z}NE"], pm[f"{z}NW"], lm[f"{z}N"])
    gmsh.model.geo.addLine(pm[f"{z}NW"], pm[f"{z}SW"], lm[f"{z}W"])
    gmsh.model.geo.addCurveLoop([lm[f"{z}S"], lm[f"{z}E"], lm[f"{z}N"], lm[f"{z}W"]], cm[f"{z}CL"])
    gmsh.model.geo.addPlaneSurface([cm[f"{z}CL"]], sm[f"{z}P"])

# add the vertical lines
for z in range(nlevels - 1):
    gmsh.model.geo.addLine(pm[f"{z}SW"], pm[f"{z+1}SW"], lm[f"{z}SW"])
    gmsh.model.geo.addLine(pm[f"{z}SE"], pm[f"{z+1}SE"], lm[f"{z}SE"])
    gmsh.model.geo.addLine(pm[f"{z}NE"], pm[f"{z+1}NE"], lm[f"{z}NE"])
    gmsh.model.geo.addLine(pm[f"{z}NW"], pm[f"{z+1}NW"], lm[f"{z}NW"])

# add the vertical faces
for z in range(nlevels - 1):
    # south face
    gmsh.model.geo.addCurveLoop([lm[f"{z}S"], lm[f"{z}SE"], -lm[f"{z+1}S"], -lm[f"{z}SW"]], cm[f"{z}S"])
    gmsh.model.geo.addPlaneSurface([cm[f"{z}S"]], sm[f"{z}S"])
    # East face
    gmsh.model.geo.addCurveLoop([lm[f"{z}E"], lm[f"{z}NE"], -lm[f"{z+1}E"], -lm[f"{z}SE"]], cm[f"{z}E"])
    gmsh.model.geo.addPlaneSurface([cm[f"{z}E"]], sm[f"{z}E"])
    # North face
    gmsh.model.geo.addCurveLoop([lm[f"{z}N"], lm[f"{z}NW"], -lm[f"{z+1}N"], -lm[f"{z}NE"]], cm[f"{z}N"])
    gmsh.model.geo.addPlaneSurface([cm[f"{z}N"]], sm[f"{z}N"])
    # West face
    gmsh.model.geo.addCurveLoop([lm[f"{z}W"], lm[f"{z}SW"], -lm[f"{z+1}W"], -lm[f"{z}NW"]], cm[f"{z}W"])
    gmsh.model.geo.addPlaneSurface([cm[f"{z}W"]], sm[f"{z}W"])

# add the volumes
for z in range(nlevels - 1):
    gmsh.model.geo.addSurfaceLoop(
        [sm[f"{z}P"], sm[f"{z+1}P"], sm[f"{z}N"], sm[f"{z}W"], sm[f"{z}S"], sm[f"{z}E"]], slm[f"{z}"]
    )
    gmsh.model.geo.addVolume([slm[f"{z}"]], vm[f"{z}"])

gmsh.model.geo.synchronize()


# create the physical volume
gmsh.model.add_physical_group(3, [vm[f"{z}"] for z in range(nlevels-1)], 1)
# create the physical surfaces
# sides
surface_index = EntityMap(100)

for z in range(nlevels-1):
    gmsh.model.add_physical_group(2, [sm[f"{z}S"],], surface_index[f"{z}S"])
    gmsh.model.add_physical_group(2, [sm[f"{z}N"],], surface_index[f"{z}N"])
    gmsh.model.add_physical_group(2, [sm[f"{z}E"],], surface_index[f"{z}E"])
    gmsh.model.add_physical_group(2, [sm[f"{z}W"],], surface_index[f"{z}W"])

# bottom
gmsh.model.add_physical_group(2, (sm[f"0P"],), surface_index["0P"])
# top
gmsh.model.add_physical_group(2, (sm[f"{nlevels-1}P"],), surface_index[f"{nlevels-1}P"])

gmsh.model.geo.synchronize()

if make_pyr:
    horizontal_planes_to_be_quads = [0,]
    for z in horizontal_planes_to_be_quads:
        # make quads
        gmsh.model.mesh.setRecombine(2, sm[f"{z}P"])
        # make structured
        gmsh.model.mesh.set_transfinite_surface(sm[f"{z}P"])

# We finally generate the mesh
gmsh.model.mesh.generate(3)

# save the mesh
gmsh.write(f"tet_pyr.msh")


gmsh.finalize()
