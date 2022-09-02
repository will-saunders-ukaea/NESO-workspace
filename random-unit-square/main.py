import pygmsh as gmsh



lc = 0.10
E = 1.0

with gmsh.geo.Geometry() as geo:
    square = geo.add_polygon([
        [0.0, 0.0],
        [E, 0.0],
        [E, E],
        [0.0, E],
    ]
    , mesh_size=lc)
    
    geo.add_physical(square, "square")
    mesh = geo.generate_mesh(dim=2, algorithm=6)

    mesh.write(f"mesh_{lc}.vtu")
    gmsh.write(f"mesh_{lc}.msh")
