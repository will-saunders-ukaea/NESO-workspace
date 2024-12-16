import gmsh
import sys
import numpy as np

# small values generate a more refined mesh
element_order = 2
lc = 0.5
validate_tolerance = 1.0e-10
# perturb_max = 0.05
# perturb_max = 0.08
perturb_max = 0.02


def make_gmsh_mesh():
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
            [
                sm[f"{z}P"],
                sm[f"{z+1}P"],
                sm[f"{z}N"],
                sm[f"{z}W"],
                sm[f"{z}S"],
                sm[f"{z}E"],
            ],
            slm[f"{z}"],
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

    # Make gmsh actually create the internal entities
    gmsh.model.mesh.create_edges()
    gmsh.model.mesh.create_faces()

    # 1 2-node line.
    # 2 3-node triangle.
    # 3 4-node quadrangle.
    # 4 4-node tetrahedron.
    # 5 8-node hexahedron.
    # 6 6-node prism.
    # 7 5-node pyramid.

    quad_ids, quad_nodes = gmsh.model.mesh.get_all_faces(4)
    edge_ids, edge_nodes = gmsh.model.mesh.get_all_edges()
    edges = {}
    for ei, ex in enumerate(edge_ids):
        edges[ex] = (edge_nodes[ei * 2], edge_nodes[ei * 2 + 1])

    def get_vertex_coords(node_id):
        return gmsh.model.mesh.get_node(node_id)[0]

    def get_all_quads():
        quads = []
        for qi, qx in enumerate(quad_ids):
            verts = quad_nodes[qi * 4 : (qi + 1) * 4 :]
            quads.append((qx, verts))

        return quads

    def validate_quad_in_plane(quad):
        verts = quad[1]

        v0 = get_vertex_coords(verts[0])
        v1 = get_vertex_coords(verts[1])
        v2 = get_vertex_coords(verts[2])
        v3 = get_vertex_coords(verts[3])

        # Make plane from v0-v1 and v0-v4 then check v3 is in the plane.
        d0 = v1 - v0
        d1 = v3 - v0
        dtest = v2 - v0
        dplane = np.cross(d0, d1)
        return abs(np.dot(dplane, dtest))

    def validate_linear_mesh():
        quads = get_all_quads()
        max_err = 0.0
        for qx in quads:
            # print(qx[0])
            # print(qx[1])
            # print(qx[2])
            err = validate_quad_in_plane(qx)
            max_err = max(err, max_err)
        print("Max error:", max_err)
        if max_err > validate_tolerance:
            raise RuntimeError("Linear validation failed")

    validate_linear_mesh()

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

        quads = get_all_quads()

        def get_vertices_only_on_triangles():
            # get the set of all vertices
            vertex_ids = set(gmsh.model.mesh.get_nodes()[0])
            # remove any vertex that is on a quad
            vertex_ids_on_quads = set()
            for qx in quads:
                for vx in qx[1]:
                    if vx in vertex_ids:
                        vertex_ids.remove(vx)
                        vertex_ids_on_quads.add(vx)
            return vertex_ids, vertex_ids_on_quads

        triangle_only_vertices, vertices_touching_quads = get_vertices_only_on_triangles()
        print("Number of triangle only vertices:", len(triangle_only_vertices))
        print("Number of vertices touching a quad:", len(vertices_touching_quads))

        def perturb_node(n):
            v = n[0]
            if not on_boundary(v):
                t = get_perturbation(v.shape)
                v += t
            return v, n[1], n[2], n[3]

        for vx in triangle_only_vertices:
            n = gmsh.model.mesh.get_node(vx)
            nn = perturb_node(n)
            gmsh.model.mesh.set_node(vx, nn[0], nn[1])

        def get_nodes_in_slicez(zmin, zmax):
            nodes = []
            vertex_ids = set(gmsh.model.mesh.get_nodes()[0])
            for vx in vertex_ids:
                z = get_vertex_coords(vx)[2]
                if (z >= zmin) and (z <= zmax):
                    nodes.append(vx)
            return nodes

        def get_nodes_in_slicexz(xmin, xmax, zmin, zmax):
            znodes = get_nodes_in_slicez(zmin, zmax)
            nodes = []
            for vx in znodes:
                x = get_vertex_coords(vx)[0]
                if (x >= xmin) and (x <= xmax):
                    nodes.append(vx)
            return nodes

        def get_nodes_in_sliceyz(ymin, ymax, zmin, zmax):
            znodes = get_nodes_in_slicez(zmin, zmax)
            nodes = []
            for vx in znodes:
                y = get_vertex_coords(vx)[1]
                if (y >= ymin) and (y <= ymax):
                    nodes.append(vx)
            return nodes

        def get_nodes_in_slicexy(xmin, xmax, ymin, ymax):
            vertex_ids = set(gmsh.model.mesh.get_nodes()[0])
            nodes0 = []
            for vx in vertex_ids:
                x = get_vertex_coords(vx)[0]
                if (x >= xmin) and (x <= xmax):
                    nodes0.append(vx)
            nodes = []
            for vx in nodes0:
                y = get_vertex_coords(vx)[1]
                if (y >= ymin) and (y <= ymax):
                    nodes.append(vx)
            return nodes

        prism_nodes = get_nodes_in_slicez(zlevels[2], zlevels[3])
        print("Number of prism nodes:", len(prism_nodes))
        for vx in prism_nodes:
            n = gmsh.model.mesh.get_node(vx)
            coords = n[0]
            if not on_boundary(coords):
                coords[2] += np.random.uniform(-perturb_max, perturb_max)
                gmsh.model.mesh.set_node(vx, coords, nn[1])

        def perturb_hex_nodes(hex_nodes, offset):
            for vx in hex_nodes:
                n = gmsh.model.mesh.get_node(vx)
                coords = n[0]
                for dimx in range(3):
                    coords[dimx] += offset[dimx]
                gmsh.model.mesh.set_node(vx, coords, nn[1])

        hex_nodes = get_nodes_in_slicexz(-0.51, -0.49, zlevels[1] - 0.01, zlevels[1] + 0.01)
        print("Number of hex nodes:", len(hex_nodes))
        offset = [0.0, 0.0, -perturb_max]
        perturb_hex_nodes(hex_nodes, offset)

        bottom_layer = 0.5 * zlevels[0] + 0.5 * zlevels[1]
        hex_nodes = get_nodes_in_slicexz(-0.51, -0.49, bottom_layer - 0.01, bottom_layer + 0.01)
        print("Number of hex nodes:", len(hex_nodes))
        offset = [0.0, 0.0, perturb_max * 0.9]
        perturb_hex_nodes(hex_nodes, offset)

        hex_nodes = get_nodes_in_sliceyz(-0.51, -0.49, zlevels[-2] - 0.01, zlevels[-2] + 0.01)
        print("Number of hex nodes:", len(hex_nodes))
        offset = [0.0, 0.0, -perturb_max * 0.95]
        perturb_hex_nodes(hex_nodes, offset)

        top_layer = 0.5 * zlevels[-1] + 0.5 * zlevels[-2]
        hex_nodes = get_nodes_in_sliceyz(-0.51, -0.49, top_layer - 0.01, top_layer + 0.01)
        print("Number of hex nodes:", len(hex_nodes))
        offset = [0.0, 0.0, perturb_max * 0.95]
        perturb_hex_nodes(hex_nodes, offset)

        hex_nodes = get_nodes_in_slicexy(0.49, 0.51, 0.49, 0.51)
        print("Number of hex nodes:", len(hex_nodes))
        offset = [perturb_max, perturb_max * 0.85, 0.0]
        perturb_hex_nodes(hex_nodes, offset)

    validate_linear_mesh()

    # compute jacobians:
    elements_blocked = gmsh.model.mesh.get_elements(3)[1]

    max_jac = -np.inf
    min_jac = np.inf

    for t in elements_blocked:
        j = gmsh.model.mesh.get_element_qualities(t)
        max_jac = max(max_jac, np.max(j))
        min_jac = min(min_jac, np.min(j))
    
    print("Min Jacobian:", min_jac)
    print("Max Jacobian:", max_jac)

    # save the mesh
    mod = "_perturbed" if perturb else ""
    order = f"_order_{element_order}" if element_order > 1 else ""
    filename = f"mixed_ref_cube_{lc}{mod}{order}.msh"
    gmsh.write(filename)
    # Launch the GUI to see the results:
    if "--visualise" in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()
    return filename


if __name__ == "__main__":
    filename = make_gmsh_mesh()
