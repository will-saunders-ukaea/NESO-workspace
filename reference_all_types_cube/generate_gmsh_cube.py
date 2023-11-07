import gmsh
import sys
import numpy as np

# small values generate a more refined mesh
element_order = 2
lc = 0.5
perturb_max = 0.05
#perturb_max = 0.08


class EntityMap:
    def __init__(self):
        self.map = {}
        self.next = 1

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

zlevels = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]


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
for z in range(5):
    gmsh.model.geo.addLine(pm[f"{z}SW"], pm[f"{z+1}SW"], lm[f"{z}SW"])
    gmsh.model.geo.addLine(pm[f"{z}SE"], pm[f"{z+1}SE"], lm[f"{z}SE"])
    gmsh.model.geo.addLine(pm[f"{z}NE"], pm[f"{z+1}NE"], lm[f"{z}NE"])
    gmsh.model.geo.addLine(pm[f"{z}NW"], pm[f"{z+1}NW"], lm[f"{z}NW"])

# add the vertical faces
for z in range(5):
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
for z in range(5):
    gmsh.model.geo.addSurfaceLoop(
        [sm[f"{z}P"], sm[f"{z+1}P"], sm[f"{z}N"], sm[f"{z}W"], sm[f"{z}S"], sm[f"{z}E"]], slm[f"{z}"]
    )
    gmsh.model.geo.addVolume([slm[f"{z}"]], vm[f"{z}"])


gmsh.model.geo.synchronize()


# create the physical volume
gmsh.model.add_physical_group(3, [vm[f"{z}"] for z in range(5)], 1)
# create the physical surfaces
# sides
gmsh.model.add_physical_group(2, [sm[f"{z}S"] for z in range(5)], 100)
gmsh.model.add_physical_group(2, [sm[f"{z}N"] for z in range(5)], 200)
gmsh.model.add_physical_group(2, [sm[f"{z}E"] for z in range(5)], 300)
gmsh.model.add_physical_group(2, [sm[f"{z}W"] for z in range(5)], 400)
# bottom
gmsh.model.add_physical_group(2, (sm[f"0P"],), 500)
# top
gmsh.model.add_physical_group(2, (sm[f"5P"],), 600)


gmsh.model.geo.synchronize()


horizontal_planes_to_be_quads = [0, 1, 4, 5]
for z in horizontal_planes_to_be_quads:
    gmsh.model.mesh.setRecombine(2, sm[f"{z}P"])
    gmsh.model.mesh.set_transfinite_surface(sm[f"{z}P"])

structured_horiztonal_planes = [2, 3]
for z in structured_horiztonal_planes:
    gmsh.model.mesh.set_transfinite_surface(sm[f"{z}P"])

vertical_planes_to_be_quads = [
    cm[f"0S"],
    cm[f"0N"],
    cm[f"0E"],
    cm[f"0W"],
    cm[f"4S"],
    cm[f"4N"],
    cm[f"4E"],
    cm[f"4W"],
    cm[f"2S"],
    cm[f"2N"],
    cm[f"2E"],
    cm[f"2W"],
]
for z in vertical_planes_to_be_quads:
    gmsh.model.mesh.setRecombine(2, z)
    gmsh.model.mesh.set_transfinite_surface(z)

volumes_to_be_structured = [
    vm[f"0"],
    vm[f"2"],
    vm[f"4"],
]
for z in volumes_to_be_structured:
    gmsh.model.mesh.set_transfinite_volume(z)




# We finally generate the mesh
gmsh.model.mesh.generate(3)

gmsh.model.mesh.setOrder(element_order)


# perturb inner points
perturb = perturb_max > 0.0
if perturb:

    def get_perturbation(shape):
        return np.random.uniform(-perturb_max, perturb_max, size=shape)

    def on_boundary(vertex):
        v0 = abs(abs(vertex[0]) - 1) < 1e-8
        v1 = abs(abs(vertex[1]) - 1) < 1e-8
        v2 = abs(abs(vertex[2]) - 1) < 1e-8
        return v0 or v1 or v2

    def perturb_node(n):
        v = n[0]
        if not on_boundary(v):
            t = get_perturbation(v.shape)
            v += t
        return v, n[1], n[2], n[3]

    vertex_ids, _, _ = gmsh.model.mesh.get_nodes()

    print("Num vertices:", len(vertex_ids))

    for vx in vertex_ids:
        n = gmsh.model.mesh.get_node(vx)
        nn = perturb_node(n)
        gmsh.model.mesh.set_node(vx, nn[0], nn[1])


# save the mesh
mod = "_perturbed" if perturb else ""
order = f"_order_{element_order}" if element_order > 1 else ""
gmsh.write(f"mixed_ref_cube_{lc}{mod}{order}.msh")
# Launch the GUI to see the results:
if "--visualise" in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
