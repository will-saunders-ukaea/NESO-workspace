import pygmsh as gmsh






lc = 0.05
E = 1.0


with gmsh.geo.Geometry() as geo:
    square = geo.add_polygon([
        [0.0, 0.0],
        [E, 0.0],
        [E, E],
        [0.0, E],
    ]
    , mesh_size=lc)

    mesh = geo.generate_mesh(dim=2, algorithm=6)

    mesh.write("mesh.vtu")
    gmsh.write("test.msh")
